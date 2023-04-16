import numpy as np
import torch

from utils import non_max_suppression, intersection_over_union, non_max_suppression_for_seq


class SeqNMS:
    def __init__(self, link_conf_thres, link_iou_thres, nms_conf_thres, nms_iou_thres, rescore):
        self.link_conf_thres = link_conf_thres
        self.link_iou_thres = link_iou_thres
        self.nms_conf_thres = nms_conf_thres
        self.nms_iou_thres = nms_iou_thres
        assert rescore == 'avg' or rescore == 'max', "Use valid rescoring methdo"
        self.rescore = rescore

        self.prev_bboxes = None
        self.prev_score = None
        self.prev_seq_length = None

    def _rm_low_conf(self, bboxes):
        """
        remove bboxes with low confidence

        :param bboxes: input bboxes without being filtered
        :return: bboxes with higher confidence
        """
        conf_bboxes = []
        for bbox in bboxes:
            if bbox[0] > self.link_conf_thres:
                conf_bboxes.append(bbox)
        return conf_bboxes

    def _cal_bbox_iou(self, bbox):
        ious = np.zeros(len(self.prev_bboxes))
        for i in range(len(self.prev_bboxes)):
            ious[i] = intersection_over_union(torch.tensor(bbox[1:]), torch.tensor(self.prev_bboxes[i][1:]),
                                              'midpoint').item()
        return ious

    def process(self, bboxes):
        """
        process a new bboxes

        :param bboxes: The input bboxes are in a list [number of boxes, 5], the second dimension contains conf score,
        x center, y center, width and height of each bbox
        :return:
        """
        prev_bboxes = []
        prev_score = []
        prev_seq_length = []

        bboxes = self._rm_low_conf(bboxes)
        if len(bboxes) == 0:
            # in case all bbox proposals have low conf score
            self.prev_bboxes = None
            self.prev_score = None
            self.prev_seq_length = None
            return bboxes

        if self.prev_bboxes is None:
            # first frame we process
            nms_bboxes = non_max_suppression(bboxes, self.nms_iou_thres, self.nms_conf_thres, "midpoint")
            for nms_bbox in nms_bboxes:
                # each nms_bboxes will be used as the start of a sequence for the next frame
                prev_bboxes.append(nms_bbox)
                # confidence score of the current nms_bboxes after NMS
                prev_score.append(nms_bbox[0])
                # we are dealing with the first frame, hence the sequence length for each bbox is 1
                prev_seq_length.append(1)
        else:
            # establish sequences
            # record the bbox ids from self.prev_bboxes that are linked to the bboxes at the current frame
            link_id = -1 * np.ones(len(bboxes), dtype=np.int16)
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                # compute IOU between each bbox of the current frame with each bbox from the previous frame
                ious_bbox = self._cal_bbox_iou(bbox)
                # based on ious_bbox identify which bbox from the previous frame has the highest IOU with the bbox of
                # interest at the current frame. If this max IOU > predefined threshold, build a connection between
                # these two bboxes
                max_iou_id = np.argmax(ious_bbox)
                if ious_bbox[max_iou_id] > self.link_iou_thres:
                    link_id[i] = max_iou_id
            # sequence selection and sequence rescoring
            # Note that each bbox from the current frame could be linked to the same bbox in the previous frame
            # from these multiple bboxes, we will only keep the one with the highest score
            # so that each bbox from the previous frame is only linked to at most 1 bbox at the current frame
            # after selecting the sequence, rescore each bbox according to either average or max
            current_score = np.asarray(bboxes)[:, 0]
            bboxes_selected = []  # bboxes after selecting the appropriate sequences
            seq_length_selected = []  # record the sequence length which each bbox is part of
            conf_score_selected = []  # confidence score of the sequence which each bbox is part of
            for prev_bbox_id in range(len(self.prev_bboxes)):
                # identify current bboxes that are linked to a specific bbox in the bbox list from the previous frame
                current_bboxes_to_prev_bbox = link_id == prev_bbox_id
                # apply NOT
                current_bboxes_to_prev_bbox_ = np.logical_not(current_bboxes_to_prev_bbox)
                # change the score for bboxes not linked to the specific bbox from the previous frame to -1 to exclude them for highest score identification
                current_score_to_prev_bbox = np.copy(current_score)
                current_score_to_prev_bbox[current_bboxes_to_prev_bbox_] = -1
                # id of current bbox with the max score to the specified prev bbox
                max_bbox_id_to_prev_bbox = np.argmax(current_score_to_prev_bbox)
                if current_score_to_prev_bbox[max_bbox_id_to_prev_bbox] == -1:
                    # indicate there is no link between bboxes in the current frame to the specified bbox in the previous frame
                    continue
                selected_bbox = bboxes[max_bbox_id_to_prev_bbox]
                # sequence length for the sequence corresponding to the current bbox + 1 to account for the current bbox
                seq_length_selected.append(self.prev_seq_length[prev_bbox_id] + 1)
                # rescore each bbox
                if self.rescore == 'max':
                    selected_bbox[0] = max(selected_bbox[0], self.prev_score[prev_bbox_id])
                    # when we use max to rescore, we record the maximum score of a sequence
                    conf_score_selected.append(selected_bbox[0])
                else:
                    # when we use avg to rescore, we record the average score of a sequence
                    conf_score_selected.append(selected_bbox[0] + self.prev_score[prev_bbox_id])
                    selected_bbox[0] = conf_score_selected[-1] / seq_length_selected[-1]
                bboxes_selected.append(selected_bbox)
            # For bboxes from the current frames that do not link to any bboxes from the previous frames
            # we add them all to bboxes_selected
            for i in range(len(link_id)):
                if link_id[i] == -1:
                    bboxes_selected.append(bboxes[i])
                    # store the conf score of bboxes that do not belong to any sequence for future rescoring
                    conf_score_selected.append(bboxes[i][0])
                    # sequence length for bboxes that do not belong to any sequence is 1
                    seq_length_selected.append(1)

            # perform nms on the bboxes that have been rescored
            bboxes_selected = np.asarray(bboxes_selected)
            bboxes_selected = np.concatenate([bboxes_selected, np.asarray(seq_length_selected).reshape(-1, 1),
                                              np.asarray(conf_score_selected).reshape(-1, 1)], axis=1)
            bboxes_selected = bboxes_selected.tolist()
            nms_bboxes = non_max_suppression_for_seq(bboxes_selected, self.nms_iou_thres, self.nms_conf_thres,
                                                     "midpoint")

            for nms_bbox in nms_bboxes:
                prev_bboxes.append(nms_bbox[:5])
                prev_seq_length.append(int(nms_bbox[5]))
                prev_score.append(nms_bbox[6])

        if len(prev_bboxes) == 0:
            # in case nms suppress all bbox, i.e., all bboxes have conf < threshold
            self.prev_bboxes = None
            self.prev_score = None
            self.prev_seq_length = None
            return prev_bboxes

        self.prev_bboxes = prev_bboxes
        self.prev_score = prev_score
        self.prev_seq_length = prev_seq_length

        return self.prev_bboxes
