import os
import argparse

import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import PascalVOC
from model import YOLOv3
from config import ANCHORS, index2class
from utils import get_eval_pred, mean_average_precision, non_max_suppression, cells_to_bboxes


def arg_parse():
    parser = argparse.ArgumentParser(description="eval YOLOv3")
    # data path
    parser.add_argument("--data_path", type=str, default="/home/xfan/Documents/Datasets")  # TODO
    # parser.add_argument("--data_path", type=str, default=os.getenv('data_path'))
    parser.add_argument("--dataset", type=str, default="VOC")
    parser.add_argument("--log_path", type=str, default="log")
    parser.add_argument("--model_name", type=str, default="eval_yolov3")
    parser.add_argument("--pretrained_model", type=str, default="log/train_yolov3")
    # model settings
    parser.add_argument("--img_sz", type=int, default=416)
    parser.add_argument("--num_cls", type=int, default=20)
    parser.add_argument("--conf_thres", type=float, default=0.8)
    parser.add_argument("--mAP_iou_thres", type=float, default=0.5)
    parser.add_argument("--nms_iou_thres", type=float, default=0.45)
    parser.add_argument("--label_iou_thres", type=float, default=0.5)
    # evaluation settings
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--train_split", action="store_false")  # TODO
    # logging settings
    parser.add_argument("--log_freq", type=int, default=10)
    return parser.parse_args()


class YOLOv3Evaluator:
    def __init__(self, args):
        self.args = args
        self.scales = [32, 16, 8]
        self.norm_anchors = []
        for scale in range(len(ANCHORS)):  # len(ANCHORS) should be 3, since the model predicts 3 scales
            norm_anchors_at_scale = []
            for anchor in range(len(ANCHORS[scale])):  # 3 anchors in each scale
                norm_anchor = [ANCHORS[scale][anchor][0] / self.args.img_sz,
                               ANCHORS[scale][anchor][1] / self.args.img_sz]  # normalize each anchor
                norm_anchors_at_scale.append(norm_anchor)
            self.norm_anchors.append(norm_anchors_at_scale)
        # anchor size in each image scale
        self.scaled_anchors = (torch.tensor(self.norm_anchors) * torch.tensor(
            [self.args.img_sz // self.scales[0], self.args.img_sz // self.scales[1],
             self.args.img_sz // self.scales[2]]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self.args.device)

        self.model = YOLOv3(num_classes=self.args.num_cls)
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
        data_path = os.path.join(self.args.data_path, self.args.dataset)
        if self.args.train_split:  # evaluate on the train split
            self.dataset = PascalVOC(data_path, True, self.norm_anchors, self.args.img_sz, self.args.num_cls,
                                     self.scales, self.args.label_iou_thres, False, False)  # TODO
        else:  # evaluate on the val split
            self.dataset = PascalVOC(data_path, False, self.norm_anchors, self.args.img_sz, self.args.num_cls,
                                     self.scales, self.args.label_iou_thres, sanity=False)  # TODO
        self.loader = DataLoader(self.dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False,
                                 drop_last=False)

        self.log_path = os.path.join(self.args.log_path, self.args.model_name)
        self.writer = SummaryWriter(os.path.join(self.log_path, "eval"))

        self.current_step = 0

        print("Begin evaluating %s" % self.args.model_name)
        print("-------------Logging Info-------------")
        print("Tensorboard event saved in: %s" % self.args.log_path)
        print("Logging frequency: %d" % self.args.log_freq)
        print("-------------Dataset Info-------------")
        print("Dataset name: %s" % self.args.dataset)
        print("Image size: %d" % self.args.img_sz)
        print("Number of classes: %d" % self.args.num_cls)
        print("Evaluate on train split: %r" % self.args.train_split)
        print("Number of evaluation images: %d" % len(self.dataset))
        print("-------------Model Info-------------")
        print("Confidence threshold: %.2f" % self.args.conf_thres)
        print("IOU threshold for generating labels: %.2f" % self.args.label_iou_thres)
        print("IOU threshold for NMS: %.2f" % self.args.nms_iou_thres)
        print("IOU threshold for mAP: %.2f" % self.args.mAP_iou_thres)

    def eval(self):
        self.model.eval()
        all_pred_boxes, all_true_boxes, tot_class_preds, correct_class, tot_noobj, correct_noobj, tot_obj, correct_obj, total_time = get_eval_pred(
            self.loader, self.model, self.args.conf_thres, self.args.nms_iou_thres, self.norm_anchors, "midpoint",
            self.args.device)
        mean_ap, cls_gt_box, cls_tp_box, cls_fp_box = mean_average_precision(all_pred_boxes, all_true_boxes,
                                                                             self.args.mAP_iou_thres, "midpoint",
                                                                             self.args.num_cls)
        self._log_result(all_pred_boxes, all_true_boxes)

        fps = len(self.dataset) / total_time
        print("mAP: %.4f" % mean_ap)
        print("GT Box for classes: ")
        print(cls_gt_box)
        print("TP Box for classes: ")
        print(cls_tp_box)
        print("FP Box for classes: ")
        print(cls_fp_box)
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
        img_bbox = img.squeeze().detach().cpu().numpy()
        img_bbox = 255 * np.transpose(img_bbox, (1, 2, 0))
        img_bbox = (img_bbox.astype(np.uint8)).copy()

        for i in range(gt_bbox.size()[0]):
            if gt_bbox[i, 1] == 1:
                x1 = (gt_bbox[i, 2] - gt_bbox[i, 4] / 2) * self.args.img_sz
                y1 = (gt_bbox[i, 3] - gt_bbox[i, 5] / 2) * self.args.img_sz
                x2 = (gt_bbox[i, 2] + gt_bbox[i, 4] / 2) * self.args.img_sz
                y2 = (gt_bbox[i, 3] + gt_bbox[i, 5] / 2) * self.args.img_sz
                cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img_bbox, index2class[int(gt_bbox[i, 0])], (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        for i in range(pred_bbox.size()[0]):
            x1 = (pred_bbox[i, 2] - pred_bbox[i, 4] / 2) * self.args.img_sz
            y1 = (pred_bbox[i, 3] - pred_bbox[i, 5] / 2) * self.args.img_sz
            x2 = (pred_bbox[i, 2] + pred_bbox[i, 4] / 2) * self.args.img_sz
            y2 = (pred_bbox[i, 3] + pred_bbox[i, 5] / 2) * self.args.img_sz
            cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_bbox, index2class[int(pred_bbox[i, 0])], (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 0, 0), 1)

        img_bbox = img_bbox / 255.0
        img_bbox = np.transpose(img_bbox, (2, 0, 1))
        img_bbox = torch.tensor(img_bbox)
        return img_bbox


if __name__ == "__main__":
    opts = arg_parse()
    evaluator = YOLOv3Evaluator(opts)
    evaluator.eval()
