# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .sparse_rcnn import SparseRCNN


_DETECTION_META_ARCHITECTURES = {"SparseRCNN":SparseRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
