import os
import cv2
import torch
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from config import class2index
from utils import iou_width_height as iou


class PascalVOC(Dataset):
    def __init__(self, data_path, is_train, norm_anchors, img_size, num_cls, down_sz, ignore_iou_thres, train_aug=True,
                 sanity=False):
        super(PascalVOC, self).__init__()
        self.data_path = data_path
        self.is_train = is_train
        self.ignore_iou_thres = ignore_iou_thres

        self.img_size = img_size
        self.num_cls = num_cls
        self.feat_sz = [self.img_size // down_sz[0], self.img_size // down_sz[1], self.img_size // down_sz[2]]

        self.norm_anchors = torch.tensor(norm_anchors[0] + norm_anchors[1] + norm_anchors[2])
        self.num_anchors = self.norm_anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

        if self.is_train:
            with open(os.path.join(data_path, "train.txt")) as f:
                self.data_list = f.readlines()
            if sanity or not train_aug:
                self.transforms = A.Compose(
                    [A.LongestMaxSize(max_size=self.img_size),
                     A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT),
                     A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
                     ToTensorV2(), ],
                    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
                )
            else:
                scale = 1.1
                self.transforms = A.Compose(
                    [A.LongestMaxSize(max_size=int(self.img_size * scale)),
                     A.PadIfNeeded(min_height=int(self.img_size * scale), min_width=int(self.img_size * scale),
                                   border_mode=cv2.BORDER_CONSTANT, ),
                     A.RandomCrop(width=self.img_size, height=self.img_size),
                     A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
                     A.OneOf([A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                              A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT), ], p=1.0, ),
                     A.HorizontalFlip(p=0.5),
                     A.Blur(p=0.1),
                     A.CLAHE(p=0.1),
                     A.Posterize(p=0.1),
                     A.ToGray(p=0.1),
                     A.ChannelShuffle(p=0.05),
                     A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
                     ToTensorV2(), ],
                    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
                )
        else:
            with open(os.path.join(data_path, "val.txt")) as f:
                self.data_list = f.readlines()
            self.transforms = A.Compose(
                [A.LongestMaxSize(max_size=self.img_size),
                 A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT),
                 A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ), ToTensorV2(), ],
                bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
            )

        for i in range(len(self.data_list)):
            self.data_list[i] = self.data_list[i].strip("\n")

        if sanity:
            self.data_list = self.data_list[0:4]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        entry = self.data_list[index]
        subset, img_id = entry.split(" ")
        img_path = os.path.join(self.data_path, subset, "JPEGImages", img_id + ".jpg")
        annotation_path = os.path.join(self.data_path, subset, "Annotations", img_id + ".xml")

        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w, _ = np.shape(image)
        annotation = ET.parse(annotation_path)
        annote_root = annotation.getroot()
        objs = annote_root.findall("object")
        bboxes = []
        for obj in objs:
            obj_name = obj.find("name")
            bndbox = obj.find("bndbox")
            cls = class2index[obj_name.text]
            xmin = int(bndbox.find("xmin").text) - 1
            ymin = int(bndbox.find("ymin").text) - 1
            xmax = int(bndbox.find("xmax").text) - 1
            ymax = int(bndbox.find("ymax").text) - 1
            x_center = (xmax + xmin) / 2
            y_center = (ymax + ymin) / 2
            width = xmax - xmin
            height = ymax - ymin
            bbox = np.array(([x_center / img_w, y_center / img_h, width / img_w, height / img_h, 1.0 * cls]))
            bboxes.append(bbox)

        augmentations = self.transforms(image=image, bboxes=bboxes)
        image = augmentations["image"]
        bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, sz, sz, 6)) for sz in self.feat_sz]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.norm_anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
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
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thres:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)
