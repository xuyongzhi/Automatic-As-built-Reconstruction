# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
import math

from maskrcnn_benchmark.layers import ROIAlignRotated3D

from .utils import cat

DEBUG = False

'''
feature map: []
POOLER_SCALES: (0.5,0.25, 0.125) is abs ratio compared with pcl.
k_min: (1, 2, 3) abs
canonical_level:
'''

class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_size, canonical_level, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_size (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_size
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        #s0 = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        s = torch.sqrt(cat([boxlist.bbox3d[:,3:5].max(dim=1)[0] for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls0 = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls0, min=self.k_min, max=self.k_max)
        levels = target_lvls.to(torch.int64) - self.k_min
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        return levels


class LevelMapper_3d(object):
    def __init__(self, scales, canonical_size):
        self.scales = torch.tensor(scales)
        self.canonical_size = canonical_size

    def __call__(self, boxlists):
        size = torch.sqrt(cat([boxlist.bbox3d[:,3:5].max(dim=1)[0] for boxlist in boxlists]))
        rate = size / self.canonical_size
        dif = torch.abs(self.scales.to(rate.device)[None,:] - rate[:,None])
        # get the smallest one within all the scales larger than rate. The dif
        # cloest to 0
        levels = torch.argmin(dif,1)
        return levels



class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlignRotated3D in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio, canonical_size, canonical_level):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlignRotated3D
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlignRotated3D(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio,
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper_3d(scales, canonical_size)
        #self.map_levels = LevelMapper(lvl_min, lvl_max, canonical_size, canonical_level)

    def convert_to_roi_format(self, boxes):
        # roialign use [center_w, center_h, roi_width, roi_height, theta]
        assert boxes[0].mode == 'yx_zb'
        boxes = [b.convert('standard') for b in boxes]
        concat_boxes = cat([b.bbox3d for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)

        rois = rois[:,[0, 2,1,3, 5,4,6, 7]] # reverse the order of x and y
        rois[:,-1]  *= 180.0/math.pi
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        if DEBUG:
          levels_num = len(x)
          print(f'\bx levels_num:{levels_num}')
          for li in range(levels_num):
            print(f"{x[li].features.shape}, {x[li].spatial_size}")
          print(f'\n boxes:')
          print(boxes)

        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        x0_features = x[0].features
        num_channels = x0_features.shape[1]
        os0,os1,os2 = self.output_size

        dtype, device = x0_features.dtype, x0_features.device
        result = torch.zeros(
            (num_rois, num_channels, os0, os1, os2),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            if DEBUG:
              print(f"\nlevel: {level}")
              print(f"f: {per_level_feature.spatial_size}")
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)

        return result

