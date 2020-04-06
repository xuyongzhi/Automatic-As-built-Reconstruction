# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 7)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        classes_names = cfg.INPUT.CLASSES
        num_classes = len(classes_names)
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        separate_classes = cfg.MODEL.SEPARATE_CLASSES
        if len(separate_classes) > 0:
          num_classes += len(separate_classes)
        self.num_classes = num_classes

        self.corner_roi = cfg.MODEL.CORNER_ROI
        self.class_specific = cfg.MODEL.CLASS_SPECIFIC

        if not self.corner_roi:
            self.cls_score = nn.Linear(representation_size, num_classes)
            if self.class_specific:
              self.bbox_pred = nn.Linear(representation_size, num_classes * 7)
            else:
              self.bbox_pred = nn.Linear(representation_size, 7 )

            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.normal_(self.bbox_pred.weight, std=0.001)
            for l in [self.cls_score, self.bbox_pred]:
                nn.init.constant_(l.bias, 0)

        else:
            self.cls_score_body = nn.Linear(representation_size, num_classes)
            if self.class_specific:
              self.bbox_pred_body = nn.Linear(representation_size, num_classes * 3) # z0, z1, thickness
              self.bbox_pred_cor = nn.Linear(representation_size, num_classes * 2) # x,y
            else:
              self.bbox_pred_body = nn.Linear(representation_size,  3) # z0, z1, thickness
              self.bbox_pred_cor = nn.Linear(representation_size, 2) # x,y

            self.bbox_corner_semantic = nn.Linear(representation_size, 8) # connection along four directions
            self.score_pred_cor = nn.Linear(representation_size, 1)


    def forward(self, x):
      if self.corner_roi:
        return self.forward_corner_box(x)
      else:
        return self.forward_centroid_box(x)

    def forward_corner_box(self, x):
        x_cor0, x_cor1, x_body = x
        scores = self.cls_score_body(x_body)
        bbox_body = self.bbox_pred_body(x_body)
        bbox_cor0 = self.bbox_pred_cor(x_cor0)
        bbox_cor1 = self.bbox_pred_cor(x_cor1)

        if self.class_specific:
          n = bbox_body.shape[0]
          bbox_cor0 = bbox_cor0.view([n, self.num_classes, 2])
          bbox_cor1 = bbox_cor1.view([n, self.num_classes, 2])
          bbox_body = bbox_body.view([n, self.num_classes, 3])
          bbox_regression_corners = torch.cat([bbox_cor0, bbox_cor1, bbox_body], 2).view([n,-1])
        else:
          bbox_regression_corners = torch.cat([bbox_cor0, bbox_cor1, bbox_body], 1)

        cor0_score = self.score_pred_cor(x_cor0)
        cor1_score = self.score_pred_cor(x_cor1)
        corner_scores = torch.cat([cor0_score, cor1_score], 1)

        bbox_corner_semantic0 = self.bbox_corner_semantic(x_cor0)
        bbox_corner_semantic1 = self.bbox_corner_semantic(x_cor1)
        corner_semantics = torch.cat([bbox_corner_semantic0, bbox_corner_semantic1],1)

        return scores, bbox_regression_corners, corner_semantics

    def forward_centroid_box(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
