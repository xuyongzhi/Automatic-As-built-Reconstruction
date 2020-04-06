# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D, merge_by_corners
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_nms_3d
from maskrcnn_benchmark.structures.boxlist_ops_3d import cat_boxlist_3d
from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from utils3d.bbox3d_ops_torch import Box3D_Torch

DEBUG = 0
SHOW_FILTER = DEBUG and 0

MERGE_BY_CORNER = 0

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self, score_thresh=0.05, nms=0.5, nms_aug_thickness=None, detections_per_img=100, box_coder=None, class_specific=True
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder3D)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder3D(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.nms_aug_thickness = nms_aug_thickness
        self.class_specific = class_specific

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList3D]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList3D]): one BoxList3D for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression, corners_semantic = x # [100*batch_size,num_class] [100*batch_size, num_classes*7]
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        size3ds = [box.size3d for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox3d for a in boxes], dim=0)

        proposals = self.box_coder.decode(
          box_regression, concat_boxes
          )

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, size3d in zip(
            class_prob, proposals, size3ds
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, size3d)
            #boxlist = boxlist.clip_to_pcl(remove_empty=False)
            if SHOW_FILTER:
              show_before_filter(boxlist, 'before filter')
            boxlist = self.filter_results(boxlist, num_classes)
            if MERGE_BY_CORNER:
              boxlist = merge_by_corners(boxlist)
            if SHOW_FILTER:
              show_before_filter(boxlist, 'after filter')
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, size3d):
        """
        Returns BoxList3D from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        if not self.class_specific:
          class_num = scores.shape[1]
          boxes = boxes.unsqueeze(1).repeat(1,class_num,1)
        boxes = boxes.reshape(-1, 7)
        scores = scores.reshape(-1)
        boxlist = BoxList3D(boxes, size3d, mode="yx_zb", examples_idxscope=None,
          constants={'prediction': True})
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox3d.reshape(-1, num_classes * 7)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 7 : (j + 1) * 7]
            boxlist_for_class = BoxList3D(boxes_j, boxlist.size3d, mode="yx_zb",
              examples_idxscope=None, constants={'prediction':True})
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms_3d(
              boxlist_for_class, nms_thresh=self.nms,
              nms_aug_thickness=self.nms_aug_thickness, score_field="scores", flag='roi_post'
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

            # debuging
            if DEBUG and False:
                inds_small_scrore = (1-inds_all[:, j]).nonzero().squeeze(1)
                scores_small_j = scores[inds_small_scrore,j]
                max_score_abandoned = scores_small_j.max()
                print(f'max_score_abandoned: {max_score_abandoned}')

        result = cat_boxlist_3d(result, per_example=False)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def show_before_filter(boxlist, msg):
  print(msg)
  boxlist.show_with_corners()
  pass

def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder3D(is_corner_roi=cfg.MODEL.CORNER_ROI,  weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    nms_aug_thickness = cfg.MODEL.ROI_HEADS.NMS_AUG_THICKNESS_Y_Z
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    class_specific = cfg.MODEL.CLASS_SPECIFIC

    postprocessor = PostProcessor(
        score_thresh, nms_thresh,
      nms_aug_thickness=nms_aug_thickness,
      detections_per_img=detections_per_img,
      box_coder=box_coder,
      class_specific=class_specific
    )
    return postprocessor
