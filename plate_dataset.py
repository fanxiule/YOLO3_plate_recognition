import os
import cv2
import torch
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from utils import iou_width_height as iou


class LicensePlateDataset(Dataset):
    def __init__(self, data_path, dataset_status, norm_anchors, img_size, down_sz, ignore_iou_thres, sanity=False):
        super(LicensePlateDataset, self).__init__()
        self.data_path = data_path
        self.ignore_iou_thres = ignore_iou_thres
        self.feat_sz = [img_size // down_sz[0], img_size // down_sz[1], img_size // down_sz[2]]
        self.norm_anchors = torch.tensor(norm_anchors[0] + norm_anchors[1] + norm_anchors[2])
        self.num_anchors = self.norm_anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // len(norm_anchors)

        if dataset_status == 0 or dataset_status == 3:
            # 0 -> training, 1 -> validation, 2 -> testing, 3 -> training + validation
            if dataset_status == 0:
                with open(os.path.join(data_path, "train.txt")) as f:
                    self.data_list = f.readlines()
            else:
                with open(os.path.join(data_path, "train_val.txt")) as f:
                    self.data_list = f.readlines()
            if sanity:
                self.transforms = A.Compose(
                    [A.LongestMaxSize(max_size=img_size),
                     A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
                     A.Normalize(),
                     ToTensorV2()],
                    bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
                )
            else:
                scale = 1.1
                self.transforms = A.Compose(
                    [A.LongestMaxSize(max_size=int(img_size * scale)),
                     A.PadIfNeeded(min_height=int(img_size * scale), min_width=int(img_size * scale),
                                   border_mode=cv2.BORDER_CONSTANT),
                     A.RandomCrop(width=img_size, height=img_size),
                     A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
                     A.OneOf([A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                              A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT), ], p=1.0),
                     A.Blur(p=0.1),
                     A.CLAHE(p=0.1),
                     A.Posterize(p=0.1),
                     A.ToGray(p=0.1),
                     A.ChannelShuffle(p=0.05),
                     A.Normalize(), ToTensorV2()],
                    bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
                )
        elif dataset_status == 1:
            with open(os.path.join(data_path, "val.txt")) as f:
                self.data_list = f.readlines()
            self.transforms = A.Compose(
                [A.LongestMaxSize(max_size=img_size),
                 A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
                 A.Normalize(), ToTensorV2()],
                bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
            )
        else:
            with open(os.path.join(data_path, "test.txt")) as f:
                self.data_list = f.readlines()
            self.transforms = A.Compose(
                [A.LongestMaxSize(max_size=img_size),
                 A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
                 A.Normalize(), ToTensorV2()],
                bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
            )

        self.data_list = [data.strip("\n") for data in self.data_list]
        self.data_list.sort()
        if sanity:
            self.data_list = self.data_list[0:4]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        entry = self.data_list[item]
        img_file, annotation_file = entry.split(", ")

        image = np.array(Image.open(img_file).convert("RGB"))
        img_h, img_w, _ = np.shape(image)
        annotation = ET.parse(annotation_file)
        annotation_root = annotation.getroot()
        objs = annotation_root.findall("object")
        bboxes = []
        for obj in objs:
            obj_name = obj.find("name")
            if obj_name.text == "licence" or obj_name.text == "license-plate":
                bndbox = obj.find("bndbox")
                xmin = np.maximum(int(bndbox.find("xmin").text) - 1, 0)
                ymin = np.maximum(int(bndbox.find("ymin").text) - 1, 0)
                xmax = int(bndbox.find("xmax").text) - 1
                ymax = int(bndbox.find("ymax").text) - 1
                x_center = (xmax + xmin) / 2
                y_center = (ymax + ymin) / 2
                width = xmax - xmin
                height = ymax - ymin
                bbox = np.array(([x_center / img_w, y_center / img_h, width / img_w, height / img_h]))
                bboxes.append(bbox)

        if len(bboxes) == 0:
            print("No bounding boxes found for %s" % img_file)
            raise RuntimeError

        augmentations = self.transforms(image=image, bboxes=bboxes)
        image = augmentations["image"]
        bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # targets store the bounding box information at different scale with respect to different anchor boxes
        targets = [torch.zeros((self.num_anchors // 3, sz, sz, 5)) for sz in self.feat_sz]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.norm_anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode="floor")
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                sz = self.feat_sz[scale_idx]
                i, j = int(sz * y), int(sz * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = sz * x - j, sz * y - i  # both between [0,1]
                    # can be greater than 1 since it's relative to cell
                    width_cell, height_cell = (width * sz, height * sz,)
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thres:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        return image, tuple(targets)
