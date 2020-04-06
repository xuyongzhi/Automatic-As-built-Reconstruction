# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from utils3d.geometric_torch import OBJ_DEF

DEBUG = True
SHOW_ANCHOR_EACH_SCALE = DEBUG and  0
CHECK_ANCHOR_STRIDES = False
if DEBUG:
  from utils3d.bbox3d_ops import Bbox3D

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        voxel_scale=20,
        sizes_3d=[[0.2,1,3], [0.5,2,3], [1,3,3]],
        yaws=(0, -1.57),
        ratios=[(1,1,1), (1,2,1)],
        use_yaws=[1,1,1],
        anchor_strides=[[8,8,729], [16,16,729], [32,32,729]],
        scene_size=[8,8,5],
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        sizes_3d = np.array(sizes_3d, dtype=np.float32)
        anchor_strides = np.array(anchor_strides)
        #levels_num = sizes_3d.shape[0]
        #for l in range(1,levels_num//2):
        #  assert sizes_3d[l,0] >= sizes_3d[l-1,0], "should start from small to large"
        #  assert anchor_strides[l,0] >= anchor_strides[l-1,0], "should start from small to large"
        assert sizes_3d.shape[1] == 3
        anchor_strides = np.array(anchor_strides, dtype=np.float32)
        assert anchor_strides.shape[1] == 3
        assert sizes_3d.shape[0] == anchor_strides.shape[0]
        yaws = np.array(yaws, dtype=np.float32).reshape([-1,1])
        ratios = np.array(ratios, dtype=np.float32)

        cell_anchors = [ generate_anchors_3d(size, yaws, ratios, uy).float()
                        for size,uy in zip(sizes_3d, use_yaws)]
        [OBJ_DEF.check_bboxes(ca, yx_zb=True) for ca in cell_anchors]
        self.anchor_num_per_loc = len(yaws) # only one size per loc
        self.voxel_scale = voxel_scale
        self.strides = torch.from_numpy(anchor_strides)
        #self.cell_anchors = BufferList(cell_anchors)
        self.cell_anchors = cell_anchors
        self.straddle_thresh = straddle_thresh
        self.anchor_mode = 'yx_zb'
        self.scene_size = torch.tensor(scene_size, dtype=torch.float)

    def num_anchors_per_location(self):
        return self.anchor_num_per_loc
        #return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, locations):
        anchors = []
        assert len(self.cell_anchors) == len(locations), "scales num not right"
        for base_anchors, location, stride in zip(
            self.cell_anchors, locations, self.strides
        ):
            anchor_centroids = (location[:,0:3].float()+0) / self.voxel_scale * stride.view(1,3)
            anchor_centroids = torch.cat([anchor_centroids, torch.zeros(anchor_centroids.shape[0],4)], 1)
            anchor_centroids = anchor_centroids.view(-1,1,7)
            base_anchors = base_anchors.view(1,-1,7)

            device = base_anchors.device
            # got standard anchors
            #flatten order [sparse_location_num, yaws_num, 7]
            anchors_scale = anchor_centroids.to(device) + base_anchors
            anchors_scale = anchors_scale.reshape(-1,7)
            anchors.append( anchors_scale )

        # CHECK_ANCHOR_STRIDES:
        if CHECK_ANCHOR_STRIDES:
          scales_num = len(anchors) // 2
          for s in range(scales_num):
            xyz_max = anchors[s][:,0:2].max(0)[0]
            er = (xyz_max / self.scene_size[0:2]).min()
            scope_min = 0.8 if s==0 else 0.7
            if er < scope_min:
              print( "CHECK_ANCHOR_STRIDES ERROR")
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              assert False
            #xyz_min = anchors[s][:,0:2].min(0)[0]
            #print(f"xyz_min:{xyz_min}\t xyz_max:{xyz_max}")

        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, points_sparse, feature_maps_sparse, targets):
        '''
          Note: all the batches are concatenated together
        '''
        #grid_sizes = [feature_map.spatial_size for feature_map in feature_maps_sparse]
        locations = [feature_map.get_spatial_locations() for feature_map in feature_maps_sparse]
        anchors_over_all_feature_maps_sparse = self.grid_anchors(locations)
        examples_idxscope = [examples_bidx_2_sizes(f.get_spatial_locations()[:,-1]) * self.anchor_num_per_loc
                              for f in feature_maps_sparse]
        anchors = [BoxList3D(a, None, self.anchor_mode, ei, {}) \
                      for a,ei in zip(anchors_over_all_feature_maps_sparse, examples_idxscope)]

        if SHOW_ANCHOR_EACH_SCALE:
            scale_num = len(anchors)
            for scale_i in range(scale_num):
              print(f'scale {scale_i} {len(anchors[scale_i])} enchors')
              min_xyz = anchors[scale_i].bbox3d[:,0:3].min(0)
              mean_xyz = anchors[scale_i].bbox3d[:,0:3].mean(0)
              max_xyz = anchors[scale_i].bbox3d[:,0:3].max(0)
              print(f'anchor ctr min: {min_xyz}')
              print(f'anchor ctr mean: {mean_xyz}')
              print(f'anchor ctr max: {max_xyz}')
              points = points_sparse[1][:,0:6].cpu().data.numpy()
              print(f'points ctr min: {points.min(0)}')
              print(f'points ctr mean: {points.mean(0)}')
              print(f'points ctr max: {points.max(0)}')

              anchors[scale_i].show(2, points_sparse[1][:,0:6], with_centroids=True, points_keep_rate=1.0)
              #anchors[scale_i].show_centroids(-1, points)
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
        return anchors


def examples_bidx_2_sizes(examples_bidx):
  batch_size = examples_bidx[-1]+1
  s = torch.tensor(0)
  e = torch.tensor(0)
  examples_idxscope = []
  for bi in range(batch_size):
    e += torch.sum(examples_bidx==bi)
    examples_idxscope.append(torch.stack([s,e]).view(1,2))
    s = e.clone()
  examples_idxscope = torch.cat(examples_idxscope, 0)
  return examples_idxscope


def make_anchor_generator(config):
    anchor_sizes_3d = config.MODEL.RPN.ANCHOR_SIZES_3D
    yaws = config.MODEL.RPN.YAWS
    ratios = config.MODEL.RPN.RATIOS
    use_yaws = config.MODEL.RPN.USE_YAWS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH
    voxel_scale = config.SPARSE3D.VOXEL_SCALE
    voxel_full_scale = config.SPARSE3D.VOXEL_FULL_SCALE
    scene_size = config.SPARSE3D.SCENE_SIZE

    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes_3d
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        voxel_scale, anchor_sizes_3d, yaws, ratios, use_yaws, anchor_stride, scene_size,  straddle_thresh
    )
    return anchor_generator


def generate_anchors_3d( size, yaws, ratios, use_yaw, centroids=np.array([[0,0,0]])):
  if use_yaw:
    return generate_anchors_3d_yaws(size, yaws, centroids)
  else:
    return generate_anchors_3d_ratio(size, ratios, centroids)


def generate_anchors_3d_ratio( size, ratios, centroids=np.array([[0,0,0]])):
    '''
      yaw == 0
      yx_zb: [xc, yc, z_bot, y_size, x_size, z_size, yaw]
      note: centroids=[0,0,0] is already xy centeroid and z bottom
    '''
    anchors = []
    yaw = np.array([0], dtype=np.float32)
    for j in range(ratios.shape[0]):
        size_j = size * ratios[j]
        for k in range(centroids.shape[0]):
          anchor = np.concatenate([centroids[k], size_j, yaw]).reshape([1,-1])
          anchors.append(anchor)
    anchors = np.concatenate(anchors, 0)
    return torch.from_numpy( anchors )

def generate_anchors_3d_yaws( size, yaws, centroids=np.array([[0,0,0]])):
    '''
      yx_zb: [xc, yc, z_bot, y_size, x_size, z_size, yaw]
      note: centroids=[0,0,0] is already xy centeroid and z bottom
    '''
    anchors = []
    for j in range(yaws.shape[0]):
        for k in range(centroids.shape[0]):
          anchor = np.concatenate([centroids[k], size, yaws[j]]).reshape([1,-1])
          anchors.append(anchor)
    anchors = np.concatenate(anchors, 0)
    return torch.from_numpy( anchors )


