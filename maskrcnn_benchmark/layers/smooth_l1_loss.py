# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, math
from utils3d.geometric_torch import limit_period

ENABLE_SYMEETRIC_CORNER = False

def parse_yaw_loss_mode(yaw_loss_mode0):
    tmp = yaw_loss_mode0.split('_')
    yaw_loss_mode1 = tmp[0]
    if len(tmp)==2:
      yaw_loss_weight = float(tmp[1])
    else:
      yaw_loss_weight = 1
    return yaw_loss_mode1, yaw_loss_weight

def get_yaw_loss(yaw_loss_mode0, input, target, anchor):
    """
    Note: target[:,-1] is the offset truth, not the yaw truth
    """
    yaw_loss_mode, yaw_loss_weight = parse_yaw_loss_mode(yaw_loss_mode0)
    assert yaw_loss_mode == 'Diff' or yaw_loss_mode == 'SinDiff'
    dif_loss = torch.abs(input[:,-1]-target[:,-1])
    if yaw_loss_mode == 'Diff':
      return dif_loss
    sin_loss = torch.sin(dif_loss)

    pred_yaw = input[:,-1] + anchor.bbox3d[:,-1]
    yaw_scope_mask = torch.abs(pred_yaw) <= math.pi/2
    yaw_loss = torch.where(yaw_scope_mask, sin_loss, dif_loss)
    yaw_loss *= yaw_loss_weight
    return yaw_loss


def smooth_l1_loss(input, target, anchor, beta=1. / 9, size_average=True, yaw_loss_mode = 'Diff'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    assert input.shape[0] == target.shape[0] == anchor.shape[0]
    assert input.shape[1] == target.shape[1] == anchor.shape[1]  == 7

    dif = torch.abs(input - target)

    dif[:,-1] = get_yaw_loss(yaw_loss_mode, input, target, anchor)

    cond = dif < beta
    loss = torch.where(cond, 0.5 * dif ** 2 / beta, dif - 0.5 * beta)

    if not ENABLE_SYMEETRIC_CORNER:
      if size_average:
          return loss.mean()
      return loss.sum()

    else:
      dif_sym_cor = torch.abs(input[:,[2,3,0,1]] - target[:,0:4])
      cond = dif_sym_cor < beta
      loss_sym_cor = torch.where(cond, 0.5 * dif_sym_cor ** 2 / beta, dif_sym_cor - 0.5 * beta)

      loss_cor1 = loss[:,0:4].sum(1)
      loss_cor2 = loss_sym_cor.sum(1) * 5
      loss_cor = torch.min(loss_cor1, loss_cor2)
      loss_zt = loss[:,4:].sum()
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      return loss_cor.sum() + loss_zt

