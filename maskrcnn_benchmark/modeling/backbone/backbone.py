# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry

from . import fpn as fpn_module
from . import resnet


@registry.BACKBONES.register("R-50-C4")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model

@registry.BACKBONES.register("Sparse-R-50-FPN")
def build_sparse_resnet_fpn_backbone(cfg):
  import sparseconvnet as scn
  dimension = 3
  residual_blocks = cfg.SPARSE3D.RESIDUAL_BLOCK
  block_reps = cfg.SPARSE3D.BLOCK_REPS
  nPlanesF = cfg.SPARSE3D.nPlanesFront
  nPlaneM = cfg.SPARSE3D.nPlaneMap
  full_scale = cfg.SPARSE3D.VOXEL_FULL_SCALE
  downsample = [cfg.SPARSE3D.KERNEL, cfg.SPARSE3D.STRIDE]
  raw_elements = cfg.INPUT.ELEMENTS
  fpn_scales = cfg.MODEL.RPN.RPN_SCALES_FROM_TOP
  roi_scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES_FROM_TOP
  voxel_scale = cfg.SPARSE3D.VOXEL_SCALE
  rpn_map_sizes = cfg.MODEL.RPN.RPN_MAP_SIZES
  rpn_3d_2d_selector = cfg.MODEL.RPN.RPN_3D_2D_SELECTOR
  bn_momentum = cfg.SOLVER.BN_MOMENTUM
  track_running_stats = cfg.SOLVER.TRACK_RUNNING_STATS

  fpn = scn.FPN_Net(full_scale, dimension, raw_elements, block_reps, nPlanesF,
                    nPlaneM = nPlaneM,
                    residual_blocks = residual_blocks,
                    fpn_scales_from_top = fpn_scales,
                    roi_scales_from_top = roi_scales,
                    downsample = downsample,
                    rpn_map_sizes = rpn_map_sizes,
                    voxel_scale = voxel_scale,
                    rpn_3d_2d_selector = rpn_3d_2d_selector,
                    bn_momentum=bn_momentum,
                    track_running_stats=track_running_stats,
                    )
  return fpn

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
