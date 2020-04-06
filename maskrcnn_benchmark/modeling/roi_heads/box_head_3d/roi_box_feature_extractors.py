# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers_3d import Pooler
import math

DEBUG = False

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()

        self.corner_roi = cfg.MODEL.CORNER_ROI

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES_SPATIAL
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        canonical_size = cfg.MODEL.ROI_BOX_HEAD.CANONICAL_SIZE
        voxel_scale = cfg.SPARSE3D.VOXEL_SCALE
        backbone_out = cfg.SPARSE3D.nPlaneMap

        pooler = Pooler(
            output_size=(resolution[0], resolution[1], resolution[2]),
            scales=scales,
            sampling_ratio=sampling_ratio,
            canonical_size=canonical_size,
            canonical_level=None
        )
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.voxel_scale = voxel_scale

        pooler_z = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION[2]
        conv3d_ = nn.Conv3d(backbone_out, representation_size,
                               kernel_size=[1,1,pooler_z], stride=[1,1,1])
        bn = nn.BatchNorm3d(representation_size, track_running_stats=cfg.SOLVER.TRACK_RUNNING_STATS)
        relu = nn.ReLU(inplace=True)
        self.conv3d = nn.Sequential(conv3d_, bn, relu)

        if not self.corner_roi:
            input_size = representation_size * resolution[0] * resolution[1]
            self.fc6 = nn.Linear(input_size, representation_size)
            self.fc7 = nn.Linear(representation_size, representation_size)

            for l in [self.fc6, self.fc7]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.kaiming_uniform_(l.weight, a=1)
                nn.init.constant_(l.bias, 0)
        else:
            roi_all_cor_body = [11, 3, 7]
            assert resolution[1] == roi_all_cor_body[0]
            input_size_cor = representation_size * resolution[0] * roi_all_cor_body[1]
            input_size_body = representation_size * resolution[0] * roi_all_cor_body[2]
            self.fc6_cor = nn.Linear(input_size_cor, representation_size)
            self.fc6_body = nn.Linear(input_size_body, representation_size)
            self.fc7_cor = nn.Linear(representation_size, representation_size)
            self.fc7_body = nn.Linear(representation_size, representation_size)

            for l in [self.fc6_cor, self.fc6_body, self.fc7_cor, self.fc7_body]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.kaiming_uniform_(l.weight, a=1)
                nn.init.constant_(l.bias, 0)

    def convert_metric_to_pixel(self, proposals0):
      #print(proposals0[0].bbox3d[:,0])
      proposals = [p.copy() for p in proposals0]
      for prop in proposals:
        prop.bbox3d[:,0:6] *= self.voxel_scale
      #print(proposals0[0].bbox3d[:,0])
      return proposals

    def forward(self, x0, proposals):
      if self.corner_roi:
        return self.forward_corner_box(x0, proposals)
      else:
        return self.forward_centroid_box(x0, proposals)

    def roi_separate(self, roi_2d):
       assert roi_2d.shape[3] == 11
       cor0 = roi_2d[:,:,:,0:3]
       body = roi_2d[:,:,:,2:9]
       cor1 = roi_2d[:,:,:,8:11]
       return [cor0, cor1, body]

    def forward_corner_box(self, x0, proposals):
      proposals = self.convert_metric_to_pixel(proposals)
      x1_ = self.pooler(x0, proposals)
      x1 = self.conv3d(x1_)
      x1s = self.roi_separate(x1.squeeze())
      x4s = []
      for i in range(3):
        x1 = x1s[i]
        x2 = x1.contiguous().view(x1.size(0), -1)
        if i < 2:
          fc6 = self.fc6_cor
          fc7 = self.fc7_cor
        else:
          fc6 = self.fc6_body
          fc7 = self.fc7_body
        x3 = F.relu(fc6(x2))
        x4 = F.relu(fc7(x3))
        x4s.append(x4)
      return x4s

    def forward_centroid_box(self, x0, proposals):
        proposals = self.convert_metric_to_pixel(proposals)
        x1_ = self.pooler(x0, proposals)
        x1 = self.conv3d(x1_)

        x2 = x1.view(x1.size(0), -1)

        x3 = F.relu(self.fc6(x2))
        x4 = F.relu(self.fc7(x3))

        if DEBUG:
          print('\nFPN2MLPFeatureExtractorN:\n')
          scale_num = len(x0)
          print(f"scale_num: {scale_num}")
          for s in range(scale_num):
            print(f"x0[{s}]: {x0[s].features.shape}, {x0[s].spatial_size}")
          print(f'x1: {x1.shape}')
          print(f'x2: {x2.shape}')
          print(f'x3: {x3.shape}')
          print(f'x4: {x4.shape}')
        return x4


def make_roi_box_feature_extractor(cfg):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
