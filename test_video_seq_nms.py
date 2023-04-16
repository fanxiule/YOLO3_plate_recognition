import os
import argparse
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import YOLOv3
from seq_nms import SeqNMS
from utils import cells_to_bboxes, non_max_suppression


def arg_parse():
    parser = argparse.ArgumentParser(description="video test YOLOv3")
    # data path
    # options:
    # ./dataset/video_test/2011_09_26_drive_0057_sync/2011_09_26/2011_09_26_drive_0057_sync/image_02/data
    # ./dataset/video_test/2011_09_26_drive_0048_sync/2011_09_26/2011_09_26_drive_0048_sync/image_02/data
    # ./dataset/video_test/2011_09_26_drive_0059_sync/2011_09_26/2011_09_26_drive_0059_sync/image_02/data
    # ./dataset/video_test/new_york_5_min_720p.mp4
    # ./dataset/video_test/0318.mp4
    parser.add_argument("--src_data", type=str,
                        default="./dataset/video_test/2011_09_26_drive_0059_sync/2011_09_26/2011_09_26_drive_0059_sync/image_02/data")
    parser.add_argument("--log_path", type=str, default="log")
    parser.add_argument("--model_name", type=str, default="detect_video_seq_nms")
    parser.add_argument("--pretrained_model", type=str, default="log/yolov3_plate_10_6_10/400")
    # model settings
    parser.add_argument("--img_sz", type=int, default=416)
    parser.add_argument("--conf_thres", type=float, default=0.7)
    parser.add_argument("--nms_iou_thres", type=float, default=0.05)
    # testing settings
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    # post processing settings
    parser.add_argument("--seq_nms", action="store_true")
    parser.add_argument("--fill_box", action="store_true")
    parser.add_argument("--seq_conf", type=float, default=0.1)
    parser.add_argument("--seq_iou", type=float, default=0.5)
    parser.add_argument("--seq_nms_rescore", type=str, choices=["max", "avg"], default="max")
    return parser.parse_args()


class VideoImageDataset(Dataset):
    # for preprocessing images data from image files
    def __init__(self, data_path, img_size):
        super(VideoImageDataset, self).__init__()
        self.data_path = data_path
        self.data_list = os.listdir(self.data_path)
        self.data_list.sort()
        self.transforms = A.Compose(
            [A.SmallestMaxSize(max_size=img_size),
             A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
             A.CenterCrop(height=img_size, width=img_size),
             A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        entry = self.data_list[item]
        entry = os.path.join(self.data_path, entry)
        image = np.array(Image.open(entry).convert("RGB"))
        augmentations = self.transforms(image=image)
        image = augmentations['image']
        return image


class VideoLoader:
    # for preprocessing images data from mp4 files
    def __init__(self, img_size):
        self.transforms = A.Compose(
            [A.SmallestMaxSize(max_size=img_size),
             A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
             A.CenterCrop(height=img_size, width=img_size),
             A.Normalize(), ToTensorV2()])

    def process_frame(self, image):
        augmentations = self.transforms(image=image)
        image = augmentations['image']
        image = torch.unsqueeze(image, 0)  # add the batch dimension
        return image


class YOLOv3VideoProcessor:
    def __init__(self, args):
        self.args = args
        self.scales = [32, 16, 8]
        self.norm_anchors = []
        anchors = [
            [(116, 90), (156, 198), (373, 326)],
            [(30, 61), (62, 45), (59, 119)],
            [(10, 13), (16, 30), (33, 23)]
        ]
        for scale in range(len(anchors)):
            norm_anchors_at_scale = []
            for anchor in range(len(anchors[scale])):  # 3 anchors in each scale
                norm_anchor = [anchors[scale][anchor][0] / self.args.img_sz,
                               anchors[scale][anchor][1] / self.args.img_sz]  # normalize each anchor
                norm_anchors_at_scale.append(norm_anchor)
            self.norm_anchors.append(norm_anchors_at_scale)
        # anchor size in each image scale
        self.scaled_anchors = (torch.tensor(self.norm_anchors) * torch.tensor(
            [self.args.img_sz // self.scales[0], self.args.img_sz // self.scales[1],
             self.args.img_sz // self.scales[2]]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self.args.device)

        self.model = YOLOv3()
        self.model = self.model.to(self.args.device)
        # load pretrained weights
        if self.args.pretrained_model is not None and os.path.exists(self.args.pretrained_model):
            checkpt_path = os.path.join(self.args.pretrained_model, "model.pth")
            assert os.path.exists(checkpt_path), "Invalid checkpoint file"
            print("Loading model %s" % self.args.pretrained_model)
            state_dict = torch.load(checkpt_path)
            self.model.load_state_dict(state_dict)
        else:
            print("Invalid pretrained model")
            raise FileNotFoundError

        if ".mp4" not in self.args.src_data:
            self.dataset = VideoImageDataset(self.args.src_data, self.args.img_sz)
            self.loader = DataLoader(self.dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False,
                                     drop_last=False)
            self.format_type = 0  # format_type = 0, data are in image form
        else:
            self.format_type = 1  # format_type = 1, data are in video form
            self.loader = VideoLoader(self.args.img_sz)

        # load post processing processor
        if self.args.seq_nms:
            self.post_processor = SeqNMS(self.args.seq_conf, self.args.seq_iou, self.args.conf_thres,
                                         self.args.nms_iou_thres, self.args.seq_nms_rescore)
        self.log_path = os.path.join(self.args.log_path, self.args.model_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.current_step = 0

        print("Begin testing %s" % self.args.model_name)
        print("-------------Logging Info-------------")
        print("Outputs saved in: %s" % self.args.log_path)
        print("-------------Dataset Info-------------")
        print("Image size: %d" % self.args.img_sz)
        print("Data source: %r" % self.args.src_data)
        print("-------------Model Info-------------")
        print("Testing device: %s" % self.args.device)
        print("Confidence threshold for NMS: %.2f" % self.args.conf_thres)
        print("IOU threshold for NMS: %.2f" % self.args.nms_iou_thres)
        print("-------------Post Processing Info-------------")
        print("Post process with Seq-NMS: %r" % self.args.seq_nms)
        print("Fill detected box with color: %r" % self.args.fill_box)
        print("Confidence threshold for Seq-NMS: %.2f" % self.args.seq_conf)
        print("IOU threshold for Seq-NMS: %.2f" % self.args.seq_iou)
        print("Rescoring method for Seq-NMS: %s" % self.args.seq_nms_rescore)

    def test(self):
        if self.format_type == 0:
            self._test_image()
        else:
            self._test_video()

    def _test_image(self):
        self.model.eval()
        elapsed_time = 0
        for batch_idx, img in enumerate(self.loader):
            start_time = time.time()
            img = img.to(self.args.device)
            with torch.no_grad():
                prediction = self.model(img)

            bboxes = []
            for i in range(3):
                S = prediction[i].shape[2]
                anchor = torch.tensor([*self.norm_anchors[i]]).to(self.args.device) * S
                boxes_scale_i = cells_to_bboxes(prediction[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes += box

            if self.args.seq_nms:
                nms_boxes = self.post_processor.process(bboxes)
            else:
                nms_boxes = non_max_suppression(bboxes, iou_threshold=self.args.nms_iou_thres,
                                                threshold=self.args.conf_thres, box_format="midpoint")

            _, img_bbox_np = self._create_pred_img(img, nms_boxes)
            elapsed_time += (time.time() - start_time)
            img_bbox_to_save = Image.fromarray(img_bbox_np)
            img_bbox_to_save.save(os.path.join(self.log_path, "%d.png" % batch_idx))
        print("FPS: %.2f" % (len(self.dataset) / elapsed_time))

    def _create_pred_img(self, img, pred_bbox):
        img = self.inv_normalize(img)
        img_bbox = img.squeeze().detach().cpu().numpy()
        img_bbox = 255 * np.transpose(img_bbox, (1, 2, 0))
        img_bbox = (img_bbox.astype(np.uint8)).copy()
        pred_bbox = torch.FloatTensor(pred_bbox)

        for i in range(pred_bbox.size()[0]):
            x1 = (pred_bbox[i, 1] - pred_bbox[i, 3] / 2) * self.args.img_sz
            y1 = (pred_bbox[i, 2] - pred_bbox[i, 4] / 2) * self.args.img_sz
            x2 = (pred_bbox[i, 1] + pred_bbox[i, 3] / 2) * self.args.img_sz
            y2 = (pred_bbox[i, 2] + pred_bbox[i, 4] / 2) * self.args.img_sz
            if self.args.fill_box:
                cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), -1)
            else:
                cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img_bbox, "Pred", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        img_bbox_torch = img_bbox / 255.0
        img_bbox_torch = np.transpose(img_bbox_torch, (2, 0, 1))
        img_bbox_torch = torch.tensor(img_bbox_torch)
        return img_bbox_torch, img_bbox

    def _test_video(self):
        self.model.eval()
        elapsed_time = 0
        cap = cv2.VideoCapture(self.args.src_data)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            if flag:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = self.loader.process_frame(img)
                start_time = time.time()
                img = img.to(self.args.device)
                with torch.no_grad():
                    prediction = self.model(img)

                bboxes = []
                for i in range(3):
                    S = prediction[i].shape[2]
                    anchor = torch.tensor([*self.norm_anchors[i]]).to(self.args.device) * S
                    boxes_scale_i = cells_to_bboxes(prediction[i], anchor, S=S, is_preds=True)
                    for _, (box) in enumerate(boxes_scale_i):
                        bboxes += box

                if self.args.seq_nms:
                    nms_boxes = self.post_processor.process(bboxes)
                else:
                    nms_boxes = non_max_suppression(bboxes, iou_threshold=self.args.nms_iou_thres,
                                                    threshold=self.args.conf_thres, box_format="midpoint")
                _, img_bbox_np = self._create_pred_img(img, nms_boxes)
                elapsed_time += (time.time() - start_time)
                img_bbox_to_save = Image.fromarray(img_bbox_np)
                img_bbox_to_save.save(os.path.join(self.log_path, "%d.png" % int(pos_frame)))
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print(str(pos_frame) + " frames")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        print("FPS: %.2f" % ((pos_frame + 1) / elapsed_time))


if __name__ == "__main__":
    opts = arg_parse()
    tester = YOLOv3VideoProcessor(opts)
    tester.test()
