import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from plate_dataset import LicensePlateDataset

ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],
    [(30, 61), (62, 45), (59, 119)],
    [(10, 13), (16, 30), (33, 23)],
]

norm_anchors = []
for scale in range(len(ANCHORS)):
    norm_anchors_at_scale = []
    for anchor in range(len(ANCHORS[scale])):
        norm_anchor = [ANCHORS[scale][anchor][0] / 416,
                       ANCHORS[scale][anchor][1] / 416]
        norm_anchors_at_scale.append(norm_anchor)
    norm_anchors.append(norm_anchors_at_scale)

plate_dataset = LicensePlateDataset("./dataset", 0, norm_anchors, 416, [32, 16, 8], 0.5, True)
img, bbox = plate_dataset.__getitem__(0)

img_np = img.detach().cpu().numpy()
img_np = np.squeeze(img_np)
img_np = np.transpose(img_np, (1, 2, 0))

bbox_13 = bbox[0].detach().cpu().numpy()
bbox_26 = bbox[1].detach().cpu().numpy()
bbox_52 = bbox[2].detach().cpu().numpy()

bbox_13_1 = np.where(bbox_13[:, :, :, 0] == 1)
bbox_13_x_center = bbox_13_1[2] + bbox_13[bbox_13_1[0], bbox_13_1[1], bbox_13_1[2], 1]
bbox_13_y_center = bbox_13_1[1] + bbox_13[bbox_13_1[0], bbox_13_1[1], bbox_13_1[2], 2]
bbox_13_x_center = bbox_13_x_center / 13 * 416
bbox_13_y_center = bbox_13_y_center / 13 * 416
bbox_13_w = bbox_13[bbox_13_1[0], bbox_13_1[1], bbox_13_1[2], 3] / 13 * 416
bbox_13_h = bbox_13[bbox_13_1[0], bbox_13_1[1], bbox_13_1[2], 4] / 13 * 416

bbox_26_1 = np.where(bbox_26[:, :, :, 0] == 1)
bbox_26_x_center = bbox_26_1[2] + bbox_26[bbox_26_1[0], bbox_26_1[1], bbox_26_1[2], 1]
bbox_26_y_center = bbox_26_1[1] + bbox_26[bbox_26_1[0], bbox_26_1[1], bbox_26_1[2], 2]
bbox_26_x_center = bbox_26_x_center / 26 * 416
bbox_26_y_center = bbox_26_y_center / 26 * 416
bbox_26_w = bbox_26[bbox_26_1[0], bbox_26_1[1], bbox_26_1[2], 3] / 26 * 416
bbox_26_h = bbox_26[bbox_26_1[0], bbox_26_1[1], bbox_26_1[2], 4] / 26 * 416

bbox_52_1 = np.where(bbox_52[:, :, :, 0] == 1)
bbox_52_x_center = bbox_52_1[2] + bbox_52[bbox_52_1[0], bbox_52_1[1], bbox_52_1[2], 1]
bbox_52_y_center = bbox_52_1[1] + bbox_52[bbox_52_1[0], bbox_52_1[1], bbox_52_1[2], 2]
bbox_52_x_center = bbox_52_x_center / 52 * 416
bbox_52_y_center = bbox_52_y_center / 52 * 416
bbox_52_w = bbox_52[bbox_52_1[0], bbox_52_1[1], bbox_52_1[2], 3] / 52 * 416
bbox_52_h = bbox_52[bbox_52_1[0], bbox_52_1[1], bbox_52_1[2], 4] / 52 * 416

x1_13 = (bbox_13_x_center - bbox_13_w / 2).astype(np.int16)
y1_13 = (bbox_13_y_center - bbox_13_h / 2).astype(np.int16)
x2_13 = (bbox_13_x_center + bbox_13_w / 2).astype(np.int16)
y2_13 = (bbox_13_y_center + bbox_13_h / 2).astype(np.int16)

x1_26 = (bbox_26_x_center - bbox_26_w / 2).astype(np.int16)
y1_26 = (bbox_26_y_center - bbox_26_h / 2).astype(np.int16)
x2_26 = (bbox_26_x_center + bbox_26_w / 2).astype(np.int16)
y2_26 = (bbox_26_y_center + bbox_26_h / 2).astype(np.int16)

x1_52 = (bbox_52_x_center - bbox_52_w / 2).astype(np.int16)
y1_52 = (bbox_52_y_center - bbox_52_h / 2).astype(np.int16)
x2_52 = (bbox_52_x_center + bbox_52_w / 2).astype(np.int16)
y2_52 = (bbox_52_y_center + bbox_52_h / 2).astype(np.int16)

if len(x1_13) != 1:
    for i in range(len(x1_13)):
        cv2.rectangle(img_np, (x1_13[i], y1_13[i]), (x2_13[i], y2_13[i]), (0, 0, 255), 2)
        cv2.rectangle(img_np, (x1_26[i], y1_26[i]), (x2_26[i], y2_26[i]), (0, 255, 0), 2)
        cv2.rectangle(img_np, (x1_52[i], y1_52[i]), (x2_52[i], y2_52[i]), (255, 0, 0), 2)
else:
    cv2.rectangle(img_np, (x1_13, y1_13), (x2_13, y2_13), (0, 0, 255), 2)
    cv2.rectangle(img_np, (x1_26, y1_26), (x2_26, y2_26), (0, 255, 0), 2)
    cv2.rectangle(img_np, (x1_52, y1_52), (x2_52, y2_52), (255, 0, 0), 2)

plt.figure()
plt.imshow(img_np)
plt.show()
print("Testing")
