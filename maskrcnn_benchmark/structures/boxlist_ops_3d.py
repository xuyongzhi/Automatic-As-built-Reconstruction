# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D, cat_boxlist_3d

from maskrcnn_benchmark.layers import nms as _box_nms
from second.pytorch.core.box_torch_ops import rotate_nms, rotate_nms_3d, multiclass_nms
from utils3d.rotate_nms_3d_torch import boxes_iou_3d
from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval

DEBUG = False

def boxlist_nms_3d(boxlist, nms_thresh, nms_aug_thickness=None, max_proposals=-1, score_field="score", flag=''):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    if nms_aug_thickness is None:
      nms_aug_thickness = [0,0]

    bn = len(boxlist)
    if flag=='rpn_post':
      #print(f'{flag}: {bn} -> {max_proposals}')
      assert max_proposals > 100, max_proposals
    elif flag=='roi_post':
      #print(f'{flag}: {bn} -> {max_proposals}')
      assert max_proposals == -1
    else:
      raise NotImplementedError
    if max_proposals<0:
      max_proposals = 500
    objectness = boxlist.get_field(score_field)

    bbox3d = boxlist.bbox3d.clone().detach()
    bbox3d[:,3:5]=  torch.clamp(bbox3d[:,3:5], min=nms_aug_thickness[0])
    bbox3d[:,5]=  torch.clamp(bbox3d[:,5], min=nms_aug_thickness[1])

    use_2d = 0
    if use_2d:
      bbox = bbox3d[:,[0,1,3,4,6]]
      rnms_f = rotate_nms
    else:
      bbox = bbox3d
      rnms_f = rotate_nms_3d
    keep = rnms_f(
              bbox,
              objectness,
              pre_max_size=2000,
              post_max_size=max_proposals,
              iou_threshold=nms_thresh, # 0.1
              flag=flag,
               )
    boxlist = boxlist[keep]
    return boxlist


def remove_small_boxes3d(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


def boxlist_iou_3d(targets, anchors, aug_thickness, criterion, only_xy=False, flag=''):
  '''
  about criterion check:
    /home/z/Research/Detection_3D/second/core/non_max_suppression/nms_gpu.py devRotateIoUEval
  '''
  assert targets.mode == 'yx_zb'
  assert anchors.mode == 'yx_zb'
  return boxes_iou_3d(targets.bbox3d, anchors.bbox3d, aug_thickness, criterion, only_xy, flag)


def test_iou_3d(bbox3d0, bbox3d1, mode):
  '''
  bbox3d: [N,7]
  '''
  boxlist3d0 = BoxList3D(bbox3d0, size3d = None, mode=mode, examples_idxscope = None, constants={})
  boxlist3d1 = BoxList3D(bbox3d1, size3d = None, mode=mode, examples_idxscope = None, constants={})
  boxlist3d0 = boxlist3d0.convert("yx_zb")
  boxlist3d1 = boxlist3d1.convert("yx_zb")

  aug_thickness = {'target':0, 'anchor':0}
  ious = boxlist_iou_3d(boxlist3d0, boxlist3d1, aug_thickness, -1)
  ious_diag = ious.diag()

  print(f"ious_diag: {ious_diag}")

  err_mask = torch.abs(ious_diag - 1) > 0.01
  err_inds = torch.nonzero(err_mask).view(-1)
  #print(f"ious:{ious}")
  print(f"err_inds: {err_inds}")
  #print(err_boxlist.bbox3d)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass


def main_test_iou_3d():
  '''
  small yaw, small thickness
  '''
  device = torch.device('cuda:0')
  #[ 43.9440, -40.0217,   0.0000,   0.0947,   2.4079,   2.7350,  -1.5549],
  #[ 43.9400, -45.1191,   0.0000,   0.0947,   2.4011,   2.7350,  -1.5550],

  bbox3d0 = torch.tensor([
   [0,0,   0.0000,   0.001,   2.,   2.,  0],
   [0,0,   0.0000,   0.01,   2.,   2.,  0],

   [0,0,   0.0000,   0.001,   2.,   2.,  np.pi/2],
   [0,0,   0.0000,   0.01,   2.,   2.,  np.pi/2],


   [0,0,   0.0000,   0.1,   2.,   2.,  np.pi/2],
   [0,0,   0.0000,   1,   2.,   2.,  np.pi/2],
      [ 2.3569,  7.0700, -0.0300,  0.0947,  1.8593,  2.7350,  0.0000],
   ],
   dtype=torch.float32
  )

  bbox3d1 = torch.tensor([
        #[ 3.9165,  2.2180, -0.0500,  0.0947,  5.0945,  2.7350, -1.5708],
        [ 7.2792,  0.2153, -0.0500,  0.0947,  1.4407,  2.7350, -1.5708],
        [ 6.5114,  0.1272, -0.0500,  0.0947,  0.2544,  2.7350,  0.0000]
    ],
   dtype=torch.float32
  )

  bbox3d1 = torch.tensor([
      [ 2.3569,  7.0700, -0.0300,  0.0947,  1.8593,  2.7350,  0.0000],
        [ 1.1548,  6.1797, -0.0300,  0.0947,  2.3096,  2.7350, -1.5708]
    ],
   dtype=torch.float32
  )

  bbox3d0 = bbox3d0.to(device)
  bbox3d1 = bbox3d1.to(device)


  test_iou_3d(bbox3d1, bbox3d1, 'yx_zb')


def main1_test_iou_3d():
  device = torch.device('cuda:0')

  bbox3d0 = torch.tensor([
      [1.2175720215e+01, 7.8515229225e+00, 5.2835583687e-02, 9.6419714391e-02,
         3.1705775261e+00, 2.7384383678e+00, 7.3978723958e-04]
    ],
   dtype=torch.float32
  )

  bbox3d1 = torch.tensor([
      [12.1804752350,  7.8437194824,  0.0490041152,  0.0947349519,
          3.1549880505,  2.7349998951,  0.0000000000]
    ],
   dtype=torch.float32
  )

  bbox3d0 = bbox3d0.to(device)
  bbox3d1 = bbox3d1.to(device)


  test_iou_3d(bbox3d1, bbox3d1, 'yx_zb')

if __name__ == '__main__':
  main1_test_iou_3d()


