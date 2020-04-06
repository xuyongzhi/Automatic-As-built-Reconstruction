# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn_sparse3d import build_rpn
from ..roi_heads.roi_heads_3d import build_roi_heads
from maskrcnn_benchmark.modeling.seperate_classifier import SeperateClassifier

DEBUG = False

class SparseRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(SparseRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.add_gt_proposals = cfg.MODEL.RPN.ADD_GT_PROPOSALS
        class_specific = cfg.MODEL.CLASS_SPECIFIC
        self.seperate_classifier = SeperateClassifier(cfg.MODEL.SEPARATE_CLASSES_ID, len(cfg.INPUT.CLASSES), class_specific, 'Detector_ROI_Heads')

    def forward(self, points, targets=None):
        """
        Arguments:
            points (list[Tensor] or ImageList): points to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        rpn_features, roi_features = self.backbone(points)
        proposals, proposal_losses = self.rpn(points, rpn_features, targets)
        if isinstance(proposals, list):
          proposals[0].clamp_size()
          proposals[1].clamp_size()
        else:
          proposals.clamp_size()
        if self.roi_heads:
            if not self.seperate_classifier.need_seperate:
              x, result, detector_losses = self.roi_heads(roi_features, proposals, targets)
            else:
              x, result, detector_losses = self.seperate_classifier.sep_roi_heads( self.roi_heads, roi_features, proposals, targets, points=points)
        else:
            # RPN-only models don't have roi_heads
            x = rpn_features
            result = proposals.seperate_examples()
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, result

        return result

