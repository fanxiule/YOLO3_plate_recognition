import os
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from plate_dataset import LicensePlateDataset
from model import YOLOv3
from utils import get_eval_pred, average_precision


def arg_parse():
    parser = argparse.ArgumentParser(description="eval YOLOv3")
    # data path
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--log_path", type=str, default="log")
    parser.add_argument("--model_name", type=str, default="eval_yolov3_test")
    parser.add_argument("--pretrained_model", type=str, default="log/yolov3_plate/400")
    # model settings
    parser.add_argument("--img_sz", type=int, default=416)
    parser.add_argument("--conf_thres", type=float, default=0.4)
    parser.add_argument("--AP_iou_thres", type=float, default=0.5)
    parser.add_argument("--nms_iou_thres", type=float, default=0.4)
    parser.add_argument("--label_iou_thres", type=float, default=0.5)
    # evaluation settings
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--split", type=int, choices=[0, 1, 2, 3], default=2)
    # logging settings
    parser.add_argument("--log_freq", type=int, default=1)
    return parser.parse_args()


class YOLOv3Evaluator:
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

        # setting up dataset
        sanity = False  # TODO
        self.dataset = LicensePlateDataset(self.args.data_path, self.args.split, self.norm_anchors, self.args.img_sz,
                                           self.scales, self.args.label_iou_thres, sanity=sanity)
        self.loader = DataLoader(self.dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False,
                                 drop_last=False)

        self.log_path = os.path.join(self.args.log_path, self.args.model_name)
        self.writer = SummaryWriter(self.log_path, "eval")

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.current_step = 0

        print("Begin evaluating %s" % self.args.model_name)
        print("-------------Logging Info-------------")
        print("Tensorboard event saved in: %s" % self.args.log_path)
        print("Logging frequency: %d" % self.args.log_freq)
        print("-------------Dataset Info-------------")
        print("Image size: %d" % self.args.img_sz)
        print("Evaluation split id: %r" % self.args.split)
        print("Number of evaluation images: %d" % len(self.dataset))
        print("-------------Model Info-------------")
        print("Evaluation device: %s" % self.args.device)
        print("Confidence threshold: %.2f" % self.args.conf_thres)
        print("IOU threshold for generating labels: %.2f" % self.args.label_iou_thres)
        print("IOU threshold for NMS: %.2f" % self.args.nms_iou_thres)
        print("IOU threshold for AP: %.2f" % self.args.AP_iou_thres)

    def eval(self):
        all_pred_boxes, all_true_boxes, tot_noobj, correct_noobj, tot_obj, correct_obj, total_time = get_eval_pred(
            self.loader, self.model, self.args.conf_thres, self.args.nms_iou_thres, self.norm_anchors, "midpoint",
            self.args.device)
        ap, gt_box, tp_box, fp_box = average_precision(all_pred_boxes, all_true_boxes, self.args.AP_iou_thres,
                                                       "midpoint")
        self._log_result(all_pred_boxes, all_true_boxes)

        fps = len(self.dataset) / total_time
        print("AP: %.4f" % ap)
        print("GT Box for classes: ", gt_box)
        print("TP Box for classes: ", tp_box)
        print("FP Box for classes: ", fp_box)
        print("FPS: %.2f" % fps)

    def _log_result(self, all_pred_boxes, all_true_boxes):
        img_id = 0
        all_pred_boxes_tensor = torch.FloatTensor(all_pred_boxes)
        all_true_boxes_tensor = torch.FloatTensor(all_true_boxes)
        while img_id < len(self.dataset):
            if img_id % self.args.log_freq == 0:
                valid_pred_box = all_pred_boxes_tensor[:, 0] == img_id
                valid_true_box = all_true_boxes_tensor[:, 0] == img_id
                valid_pred_box = all_pred_boxes_tensor[valid_pred_box]
                valid_true_box = all_true_boxes_tensor[valid_true_box]
                img, _ = self.dataset.__getitem__(img_id)
                img_bbox = self._create_pred_imgs(img, valid_pred_box[:, 1:], valid_true_box[:, 1:])
                self.writer.add_image("Sample_prediction", img_bbox, img_id)
            img_id += 1

    def _create_pred_imgs(self, img, pred_bbox, gt_bbox):
        img = self.inv_normalize(img)
        img_bbox = img.squeeze().detach().cpu().numpy()
        img_bbox = 255 * np.transpose(img_bbox, (1, 2, 0))
        img_bbox = (img_bbox.astype(np.uint8)).copy()

        for i in range(gt_bbox.size()[0]):
            if gt_bbox[i, 0] == 1:
                x1 = (gt_bbox[i, 1] - gt_bbox[i, 3] / 2) * self.args.img_sz
                y1 = (gt_bbox[i, 2] - gt_bbox[i, 4] / 2) * self.args.img_sz
                x2 = (gt_bbox[i, 1] + gt_bbox[i, 3] / 2) * self.args.img_sz
                y2 = (gt_bbox[i, 2] + gt_bbox[i, 4] / 2) * self.args.img_sz
                cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img_bbox, "GT", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        for i in range(pred_bbox.size()[0]):
            x1 = (pred_bbox[i, 1] - pred_bbox[i, 3] / 2) * self.args.img_sz
            y1 = (pred_bbox[i, 2] - pred_bbox[i, 4] / 2) * self.args.img_sz
            x2 = (pred_bbox[i, 1] + pred_bbox[i, 3] / 2) * self.args.img_sz
            y2 = (pred_bbox[i, 2] + pred_bbox[i, 4] / 2) * self.args.img_sz
            cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_bbox, "Pred", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        img_bbox = img_bbox / 255.0
        img_bbox = np.transpose(img_bbox, (2, 0, 1))
        img_bbox = torch.tensor(img_bbox)
        return img_bbox


if __name__ == "__main__":
    opts = arg_parse()
    evaluator = YOLOv3Evaluator(opts)
    evaluator.eval()
