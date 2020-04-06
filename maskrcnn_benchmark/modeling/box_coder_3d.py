# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

from utils3d.bbox3d_ops_torch import Box3D_Torch, box_dif
import torch
from second.pytorch.core.box_torch_ops import second_box_encode, second_box_decode, second_corner_box_encode, second_corner_box_decode
from utils3d.geometric_torch import limit_period

class BoxCoder3D(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, is_corner_roi, weights):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.is_corner_roi = is_corner_roi
        self.smooth_dim = True
        if weights is None:
            weights=(1.0,)*7
        weights = torch.tensor(weights).view(1,7)
        self.weights = weights
        if not self.smooth_dim:
          # Prevent sending too large values into torch.exp()
          bbox_xform_clip = math.log(1000. / 1)
        else:
          bbox_xform_clip = 10000. / 1
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, targets, anchors):
        if self.is_corner_roi:
          return self.encode_corner_box(targets, anchors)
        else:
          return self.encode_centroid_box(targets, anchors)

    def decode(self, box_encodings, anchors):
        if self.is_corner_roi:
          return  self.decode_corner_box(box_encodings, anchors)
        else:
          return  self.decode_centroid_box(box_encodings, anchors)

    def encode_centroid_box(self, targets, anchors):
        box_encodings = second_box_encode(targets, anchors, smooth_dim=self.smooth_dim)
        # yaw diff in [-pi/2, pi/2]
        box_encodings[:,-1] = limit_period(box_encodings[:,-1], 0.5, math.pi)
        box_encodings = box_encodings * self.weights.to(box_encodings.device)
        return box_encodings

    def decode_centroid_box(self, box_encodings, anchors):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        assert box_encodings.shape[0] == anchors.shape[0]
        assert anchors.shape[1] == 7
        num_classes = int(box_encodings.shape[1]/7)
        if num_classes != 1:
          num_loc = box_encodings.shape[0]
          box_encodings = box_encodings.view(-1, 7)
          anchors = anchors.view(num_loc,1,7)
          anchors = anchors.repeat(1,num_classes,1).view(-1,7)

        box_encodings = box_encodings / self.weights.to(box_encodings.device)
        box_encodings[:,3:6] = torch.clamp(box_encodings[:,3:6], max=self.bbox_xform_clip)
        boxes_decoded = second_box_decode(box_encodings, anchors, smooth_dim=self.smooth_dim)
        # yaw diff in [-pi/2, pi/2]
        boxes_decoded[:,-1] = limit_period(boxes_decoded[:,-1], 0.5, math.pi)

        if num_classes != 1:
          boxes_decoded = boxes_decoded.view(-1,num_classes*7)

        return boxes_decoded

    def encode_corner_box(self, targets_yxzb, anchors_yxzb):
        '''
        input: boxes of yx_zb
        '''
        targets = Box3D_Torch.from_yxzb_to_2corners(targets_yxzb)
        anchors = Box3D_Torch.from_yxzb_to_2corners(anchors_yxzb)

        box_encodings = second_corner_box_encode(targets, anchors, smooth_dim=self.smooth_dim)
        box_encodings = box_encodings * self.weights.to(box_encodings.device)
        return box_encodings

    def decode_corner_box(self, box_encodings_2corners, proposals_yxzb):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            box_encodings_2corners (Tensor): encoded boxes of 2 corners format
            boxes (Tensor): reference boxes of yx_zb format
        """
        assert box_encodings_2corners.shape[0] == proposals_yxzb.shape[0]
        assert proposals_yxzb.shape[1] == 7
        proposals = Box3D_Torch.from_yxzb_to_2corners(proposals_yxzb)

        num_classes = int(box_encodings_2corners.shape[1]/7)
        if num_classes != 1:
          num_loc = box_encodings_2corners.shape[0]
          box_encodings_2corners = box_encodings_2corners.view(-1, 7)
          proposals = proposals.view(num_loc,1,7)
          proposals = proposals.repeat(1,num_classes,1).view(-1,7)

        box_encodings_2corners = box_encodings_2corners / self.weights.to(box_encodings_2corners.device)
        box_encodings_2corners[:,3:6] = torch.clamp(box_encodings_2corners[:,3:6], max=self.bbox_xform_clip)
        boxes_decoded = second_corner_box_decode(box_encodings_2corners, proposals, smooth_dim=self.smooth_dim)

        boxes_decoded_yxzb = Box3D_Torch.from_2corners_to_yxzb(boxes_decoded)

        if num_classes != 1:
          boxes_decoded_yxzb = boxes_decoded_yxzb.view(-1,num_classes*7)

        return boxes_decoded_yxzb

def gen_rand_box_yxzb():
    boxes = torch.rand(2,7, dtype=torch.float)
    boxes[:,-1] = ( boxes[:,-1] - 0.5 ) * math.pi
    boxes[:,4] += 2
    boxes[:,5] += 1
    return boxes

def test():
  boxcoder = BoxCoder3D()
  targets_yxzb0 = gen_rand_box_yxzb()
  anchors_yxzb0 = gen_rand_box_yxzb()

  regression0 = boxcoder.encode( targets_yxzb0, anchors_yxzb0 )
  targets_yxzb1 = boxcoder.decode(regression0, anchors_yxzb0)
  dif1 = box_dif(targets_yxzb0, targets_yxzb1)
  print(f'dif: {dif1}')
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass


if __name__ == '__main__':
  test()


