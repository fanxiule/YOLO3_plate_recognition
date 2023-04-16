import os
import argparse
import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from plate_dataset import LicensePlateDataset
from model import YOLOv3
from loss import YoloLoss
from utils import get_eval_pred, average_precision, non_max_suppression, cells_to_bboxes


def arg_parse():
    # data path
    parser = argparse.ArgumentParser(description="train YOLOv3")
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--log_path", type=str, default="log")
    parser.add_argument("--model_name", type=str, default="yolov3_sanity")
    parser.add_argument("--pretrained_model", type=str, default=None)
    # model settings
    parser.add_argument("--img_sz", type=int, default=416)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf_thres", type=float, default=0.4)
    parser.add_argument("--AP_iou_thres", type=float, default=0.5)
    parser.add_argument("--nms_iou_thres", type=float, default=0.4)
    parser.add_argument("--label_iou_thres", type=float, default=0.5)
    # optimization settings
    parser.add_argument("--train_split", type=int, choices=[0, 3], default=0)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler_step", type=int, default=200)
    parser.add_argument("--scheduler_rate", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    # Loss function settings
    parser.add_argument("--lambda_box", type=int, default=10)
    parser.add_argument("--lambda_obj", type=float, default=1)
    parser.add_argument("--lambda_noobj", type=float, default=10)
    # logging settings
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=200)
    return parser.parse_args()


class YOLOv3Trainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        self.scales = [32, 16, 8]
        self.norm_anchors = []
        anchors = [
            [(116, 90), (156, 198), (373, 326)],
            [(30, 61), (62, 45), (59, 119)],
            [(10, 13), (16, 30), (33, 23)]
        ]
        for scale in range(len(anchors)):
            norm_anchors_at_scale = []
            for anchor in range(len(anchors[scale])):
                norm_anchor = [anchors[scale][anchor][0] / self.args.img_sz,
                               anchors[scale][anchor][1] / self.args.img_sz]
                norm_anchors_at_scale.append(norm_anchor)
            self.norm_anchors.append(norm_anchors_at_scale)
        self.scaled_anchors = (torch.tensor(self.norm_anchors) * torch.tensor(
            [self.args.img_sz // self.scales[0], self.args.img_sz // self.scales[1],
             self.args.img_sz // self.scales[2]]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self.args.device)

        self.model = YOLOv3()
        self.model = self.model.to(self.args.device)
        self.model.init_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.args.scheduler_step, self.args.scheduler_rate)
        self.loss = YoloLoss(self.args.lambda_box, self.args.lambda_obj, self.args.lambda_noobj)

        if self.args.pretrained_model is not None:
            print("Loading pretrained weights from %s" % self.args.pretrained_model)
            model_checkpt = os.path.join(self.args.pretrained_model, "model.pth")
            optim_checkpt = os.path.join(self.args.pretrained_model, "adam.pth")
            if os.path.exists(model_checkpt):
                model_state_dict = torch.load(model_checkpt)
                self.model.load_state_dict(model_state_dict)
                print("Loaded model weights")
            else:
                print("Use random model weights")

            if os.path.exists(optim_checkpt):
                optim_state_dict = torch.load(optim_checkpt)
                self.optimizer.load_state_dict(optim_state_dict)
                print("Loaded optimizer weights")
            else:
                print("Use random optimizer weights")
        else:
            print("Use random model and optimizer weights")

        sanity = False
        self.train_dataset = LicensePlateDataset(self.args.data_path, self.args.train_split, self.norm_anchors,
                                                 self.args.img_sz, self.scales, self.args.label_iou_thres,
                                                 sanity=sanity)
        self.val_dataset = LicensePlateDataset(self.args.data_path, 1, self.norm_anchors, self.args.img_sz,
                                               self.scales, self.args.label_iou_thres, sanity=sanity)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_sz,
                                       num_workers=self.args.num_workers, pin_memory=True, shuffle=True,
                                       drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_sz, num_workers=self.args.num_workers,
                                     pin_memory=True, shuffle=False, drop_last=False)
        self.val_iter = iter(self.val_loader)

        self.log_path = os.path.join(self.args.log_path, self.args.model_name)
        self.train_writer = SummaryWriter(os.path.join(self.log_path, "train"))
        self.val_writer = SummaryWriter(os.path.join(self.log_path, "val"))

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.current_epoch = 0
        self.current_step = 0

        print("Begin training %s" % self.args.model_name)
        print("-------------Logging Info-------------")
        print("Checkpoints and log saved in: %s" % self.args.log_path)
        print("Checkpoint save frequency: %d" % self.args.save_freq)
        print("Mini-batch is evaluated at frequency: %d" % self.args.log_freq)
        print("-------------Model Info-------------")
        print("Image size: %d" % self.args.img_sz)
        print("Confidence threshold: %.2f" % self.args.conf_thres)
        print("IOU threshold for generating labels: %.2f" % self.args.label_iou_thres)
        print("IOU threshold for NMS: %.2f" % self.args.nms_iou_thres)
        print("IOU threshold for AP: %.2f" % self.args.AP_iou_thres)
        print("-------------Optimization Info-------------")
        print("Training device: %s" % self.args.device)
        print("Training set identification: %d" % self.args.train_split)
        print("Number of training images: %d" % len(self.train_dataset))
        print("Number of validation images: %d" % len(self.val_dataset))
        print("Batch size: %d" % self.args.batch_sz)
        print("Number of epochs: %d" % self.args.num_epochs)
        print("Initial learning rate: %.5f" % self.args.lr)
        print("Scheduler step: %d" % self.args.scheduler_step)
        print("Scheduler rate: %.2f" % self.args.scheduler_rate)
        print("-------------Loss Info-------------")
        print("Lambda box: %.2f" % self.args.lambda_box)
        print("Lambda obj: %.2f" % self.args.lambda_obj)
        print("Lambda noobj: %.2f" % self.args.lambda_noobj)

    def train(self):
        print("---------------Start training---------------")
        for self.current_epoch in range(self.args.num_epochs):
            print("Training epoch %d" % (self.current_epoch + 1))
            self.train_epoch()
            self.scheduler.step()
            if (self.current_epoch + 1) % self.args.save_freq == 0:
                self.val()
                save_folder = os.path.join(self.log_path, "%d" % (self.current_epoch + 1))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                checkpt_file = os.path.join(save_folder, "model.pth")
                optim_file = os.path.join(save_folder, "adam.pth")
                torch.save(self.model.state_dict(), checkpt_file)
                torch.save(self.optimizer.state_dict(), optim_file)

    def train_epoch(self):
        self.model.train()
        for batch_idx, (imgs, targets) in enumerate(self.train_loader):
            self.current_step += 1
            imgs = imgs.to(self.args.device)
            imgs = imgs
            target0, target1, target2 = (
                targets[0].to(self.args.device), targets[1].to(self.args.device), targets[2].to(self.args.device))
            predictions = self.model(imgs)
            loss = (self.loss(predictions[0], torch.clone(target0), self.scaled_anchors[0])
                    + self.loss(predictions[1], torch.clone(target1), self.scaled_anchors[1])
                    + self.loss(predictions[2], torch.clone(target2), self.scaled_anchors[2]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.current_step % self.args.log_freq == 0:
                print("Training Loss: %.4f" % loss)
                bboxes = []  # only consider the first sample in the batch
                for i in range(3):
                    S = predictions[i].shape[2]
                    anchor = torch.tensor([*self.norm_anchors[i]]).to(self.args.device) * S
                    boxes_scale_i = cells_to_bboxes(predictions[i][0:1, ...], anchor, S=S, is_preds=True)
                    bboxes += boxes_scale_i[0]
                pred_bboxes = non_max_suppression(bboxes, self.args.nms_iou_thres, self.args.conf_thres, "midpoint")
                gt_bboxes = cells_to_bboxes(target0[0:1], self.scaled_anchors[0], self.args.img_sz // self.scales[0],
                                            False)
                img_bbox = self._create_pred_imgs(imgs[0], pred_bboxes, gt_bboxes[0])
                self.train_writer.add_image("Sample_prediction", img_bbox, self.current_step)
                self.train_writer.add_scalar("Loss", loss, self.current_step)
                self.val_mini_batch()

    def val(self):
        all_pred_boxes, all_true_boxes, tot_noobj, correct_noobj, tot_obj, correct_obj, _ = get_eval_pred(
            self.val_loader, self.model, self.args.conf_thres, self.args.nms_iou_thres, self.norm_anchors, "midpoint",
            self.args.device)
        self._cal_cls_obj_acc(tot_noobj, correct_noobj, tot_obj, correct_obj)
        ap, _, _, _ = average_precision(all_pred_boxes, all_true_boxes, self.args.AP_iou_thres, "midpoint")
        print("AP: %.2f" % ap)
        self.val_writer.add_scalar("AP", ap, self.current_step)

    def _cal_cls_obj_acc(self, tot_noobj, correct_noobj, tot_obj, correct_obj):
        eps = 1e-16
        noobj_acc = (correct_noobj / (tot_noobj + eps)) * 100
        obj_acc = (correct_obj / (tot_obj + eps)) * 100
        print("No obj accuracy: %.2f" % noobj_acc)
        print("Obj accuracy: %.2f" % obj_acc)
        self.val_writer.add_scalar("No_Obj_Accuracy", noobj_acc, self.current_step)
        self.val_writer.add_scalar("Obj_Accuracy", obj_acc, self.current_step)

    def val_mini_batch(self):
        self.model.eval()
        try:
            imgs, targets = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            imgs, targets = self.val_iter.next()
        imgs = imgs.to(self.args.device)
        target0, target1, target2 = (
            targets[0].to(self.args.device), targets[1].to(self.args.device), targets[2].to(self.args.device))
        with torch.no_grad():
            predictions = self.model(imgs)
            loss = (self.loss(predictions[0], torch.clone(target0), self.scaled_anchors[0])
                    + self.loss(predictions[1], torch.clone(target1), self.scaled_anchors[1])
                    + self.loss(predictions[2], torch.clone(target2), self.scaled_anchors[2]))
        bboxes = []  # only consider the first sample in the batch
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*self.norm_anchors[i]]).to(self.args.device) * S
            boxes_scale_i = cells_to_bboxes(predictions[i][0:1, ...], anchor, S=S, is_preds=True)
            bboxes += boxes_scale_i[0]
        pred_bboxes = non_max_suppression(bboxes, self.args.nms_iou_thres, self.args.conf_thres, "midpoint")
        gt_bboxes = cells_to_bboxes(target0[0:1], self.scaled_anchors[0], self.args.img_sz // self.scales[0], False)
        img_bbox = self._create_pred_imgs(imgs[0], pred_bboxes, gt_bboxes[0])
        print("Validation loss: %.4f" % loss)
        self.val_writer.add_scalar("Loss", loss, self.current_step)
        self.val_writer.add_image("Sample_prediction", img_bbox, self.current_step)
        self.model.train()

    def _create_pred_imgs(self, img, pred_bbox, gt_bbox):
        img = self.inv_normalize(img)
        img_bbox = img.squeeze().detach().cpu().numpy()
        img_bbox = 255 * np.transpose(img_bbox, (1, 2, 0))
        img_bbox = (img_bbox.astype(np.uint8)).copy()

        for i in range(len(gt_bbox)):
            if gt_bbox[i][0] == 1:
                x1 = (gt_bbox[i][1] - gt_bbox[i][3] / 2) * self.args.img_sz
                y1 = (gt_bbox[i][2] - gt_bbox[i][4] / 2) * self.args.img_sz
                x2 = (gt_bbox[i][1] + gt_bbox[i][3] / 2) * self.args.img_sz
                y2 = (gt_bbox[i][2] + gt_bbox[i][4] / 2) * self.args.img_sz
                cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img_bbox, "GT", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        for i in range(len(pred_bbox)):
            x1 = (pred_bbox[i][1] - pred_bbox[i][3] / 2) * self.args.img_sz
            y1 = (pred_bbox[i][2] - pred_bbox[i][4] / 2) * self.args.img_sz
            x2 = (pred_bbox[i][1] + pred_bbox[i][3] / 2) * self.args.img_sz
            y2 = (pred_bbox[i][2] + pred_bbox[i][4] / 2) * self.args.img_sz
            cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_bbox, "Pred", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        img_bbox = img_bbox / 255.0
        img_bbox = np.transpose(img_bbox, (2, 0, 1))
        img_bbox = torch.tensor(img_bbox)
        return img_bbox


if __name__ == "__main__":
    opts = arg_parse()
    trainer = YOLOv3Trainer(opts)
    trainer.train()
