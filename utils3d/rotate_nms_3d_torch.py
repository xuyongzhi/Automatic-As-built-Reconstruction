import torch
import numpy as np
from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval

DEBUG = 1

def iou_one_dim(targets_z, anchors_z):
    '''
    For ceiling, and floor: z size of target is small, augment to 1
    '''
    #targets_z[:,1] = torch.clamp(targets_z[:,1], min=0.8)
    #anchors_z[:,1] = torch.clamp(anchors_z[:,1], min=0.8) # aug proposal for ROI input as well

    anchors_z[:,1] = anchors_z[:,0] + anchors_z[:,1]
    targets_z[:,1] = targets_z[:,0] + targets_z[:,1]
    targets_z = targets_z.unsqueeze(1)
    anchors_z = anchors_z.unsqueeze(0)
    overlap = torch.min(anchors_z[:,:,1], targets_z[:,:,1]) - torch.max(anchors_z[:,:,0], targets_z[:,:,0])
    common = torch.max(anchors_z[:,:,1], targets_z[:,:,1]) - torch.min(anchors_z[:,:,0], targets_z[:,:,0])
    iou_z = overlap / common
    return iou_z

def boxes_iou_3d(targets_bbox3d, anchors_bbox3d, aug_thickness=None, criterion=-1, only_xy=False, flag=''):
  '''
  about criterion check:
    /home/z/Research/Detection_3D/second/core/non_max_suppression/nms_gpu.py devRotateIoUEval

  # implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
  # with slight modifications
  '''

  if DEBUG:
    only_xy = True
  if flag == 'rpn_label_generation':
    assert aug_thickness['anchor_Y'] == 0
    assert aug_thickness['target_Y'] >= 0.3
  elif flag == 'roi_label_generation':
    assert aug_thickness['anchor_Y'] >= 0.3
    assert aug_thickness['target_Y'] >= 0.3
  elif flag == 'eval':
    assert aug_thickness['anchor_Y'] <= 0.3
    assert aug_thickness['target_Y'] <= 0.3
  elif flag == 'rpn_post' or flag == 'roi_post':
    assert aug_thickness is None
  else:
    print(flag)
    print(aug_thickness)
    raise NotImplementedError

  if aug_thickness is None:
    ma = 0.0
    aug_thickness = {'target_Y':ma, 'target_Z':ma, 'anchor_Y':ma, 'anchor_Z':ma}

  #print(f'{flag}\n{aug_thickness}\n')

  targets_bbox3d = targets_bbox3d.clone().detach()
  anchors_bbox3d = anchors_bbox3d.clone().detach()

  targets_bbox3d[:,3] = torch.clamp(targets_bbox3d[:,3], min=aug_thickness['target_Y'])
  anchors_bbox3d[:,3] = torch.clamp(anchors_bbox3d[:,3], min=aug_thickness['anchor_Y'])
  targets_bbox3d[:,5] = torch.clamp(targets_bbox3d[:,5], min=aug_thickness['target_Z'])
  anchors_bbox3d[:,5] = torch.clamp(anchors_bbox3d[:,5], min=aug_thickness['anchor_Z'])

  iouz = iou_one_dim(targets_bbox3d[:,[2,5]], anchors_bbox3d[:,[2,5]])

  cuda_index = targets_bbox3d.device.index
  anchors_2d = anchors_bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()
  targets_2d = targets_bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()

  #print(f"targets yaw : {targets_2d[:,-1].min()} , {targets_2d[:,-1].max()}")
  #print(f"anchors yaw : {anchors_2d[:,-1].min()} , {anchors_2d[:,-1].max()}")

  # aug thickness. When thickness==0, iou is wrong
  #targets_2d[:,2] = np.clip(targets_2d[:,2], a_min=aug_thickness['target'], a_max=None)
  #anchors_2d[:,2] = np.clip(anchors_2d[:,2], a_min=aug_thickness['anchor'], a_max=None)

  #aug_th_mask = (targets_2d[:,2] < 0.3).astype(np.float32)
  #targets_2d[:,2] += aug_thickness['target'] * aug_th_mask  # 0.25
  #aug_th_mask = (anchors_2d[:,2] < 0.3).astype(np.float32)
  #anchors_2d[:,2] += aug_thickness['anchor'] * aug_th_mask

  # criterion=1: use targets_2d as ref
  iou2d = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=criterion, device_id=cuda_index)
  iou2d = torch.from_numpy(iou2d)
  iou2d = iou2d.to(targets_bbox3d.device)

  if only_xy:
      iou3d = iou2d
  else:
      iou3d = iou2d * iouz

  if DEBUG and flag=='eval' and False:
      if iou3d.max() < 1:
          return iou3d

      mask = iou3d == iou3d.max()
      t_i, a_i = torch.nonzero(mask)[0]
      t = targets[t_i]
      a = anchors[a_i]
      print(f"max iou: {iou3d.max()}")
      a.show_together(t)

      if iou3d.max() > 1:
          torch.set_printoptions(precision=16)
          print(a.bbox3d)
          print(t.bbox3d)

          #np.set_printoptions(precision=10)
          #print(a.bbox3d.cpu().data.numpy())
          #print(t.bbox3d.cpu().data.numpy())

          areas = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=3, device_id=cuda_index)
          ious0 = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=0, device_id=cuda_index)
          ious1 = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=1, device_id=cuda_index)
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          areas_max = areas[t_i, a_i]
          import pdb; pdb.set_trace()  # XXX BREAKPOINT


      iou_preds = iou3d.max(0)[0]
      mask = iou3d == iou_preds.min()
      t_i, a_i = torch.nonzero(mask)[0]
      t = targets[t_i]
      a = anchors[a_i]
      print(f"min pred iou: {iou_preds.min()}")
      a.show_together(t)
      anchors.show_highlight([a_i])
      targets.show_highlight([t_i])
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  return iou3d
