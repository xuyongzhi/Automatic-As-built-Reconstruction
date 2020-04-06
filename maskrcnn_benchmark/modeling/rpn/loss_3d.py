# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_iou_3d, cat_boxlist_3d
import numpy as np

from data3d.dataset_metas import DSET_METAS

DEBUG = True
SHOW_POS_ANCHOR_IOU_SAME_LOC = DEBUG and 0
CHECK_MATCHER = DEBUG and 0

SHOW_IGNORED_ANCHOR = DEBUG and 0
SHOW_POS_NEG_ANCHORS = DEBUG and 1

SHOW_PRED_POS_ANCHORS = DEBUG and  False
CHECK_REGRESSION_TARGET_YAW = False

def check_matcher(target, anchor, match_quality_matrix, matched_idxs, dset_metas):
  ONLY_MISSED_TARGET = False

  num_gt = target.bbox3d.shape[0]
  labels = target.get_field('labels').cpu().data.numpy().astype(np.int)
  classes = [dset_metas.label_2_class[l] for l in labels]
  for j in range(num_gt):
    ious_j = match_quality_matrix[j]
    mathched_inds_j = torch.nonzero(matched_idxs == j).squeeze(1)
    iou_m_j = ious_j[mathched_inds_j]
    a_m_j = anchor[mathched_inds_j]

    if ONLY_MISSED_TARGET and mathched_inds_j.shape[0] != 0:
        continue

    print(f'\n{j}-th target The checked {classes[j]}\n')
    target.show_highlight([j])

    print(f"matched iou: {iou_m_j}")
    a_m_j.show__together(target[j], points=anchor.centroids() )

    ious_j_top, ids_top_j = ious_j.topk(5)
    print(f"top ious:{ious_j_top}")
    matched_ids_top = matched_idxs[ids_top_j]
    print(f'matched_ids_top: {matched_ids_top}')

    for k in range(len(ids_top_j)):
      a_j_top = anchor[ids_top_j[k]]
      the_matched_idx = matched_idxs[ids_top_j[k]]
      print(f"\n\niou:{ious_j_top[k]}, matched idx:{the_matched_idx}")
      a_j_top.show__together(target[j], points=anchor.centroids())
      if the_matched_idx >= 0 and the_matched_idx!=j:
        iou_another = match_quality_matrix[the_matched_idx, ids_top_j[k]]
        print(f'this anchor matched with another target with iou: {iou_another}')
        a_j_top.show__together(target[[j, the_matched_idx]], points=anchor.centroids())
    pass
  pass

class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, yaw_loss_mode, aug_thickness, dset_metas, IoU_criteria):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.yaw_loss_mode = yaw_loss_mode
        self.aug_thickness = aug_thickness
        self.dset_metas = dset_metas
        self.IoU_criteria = IoU_criteria

    def match_targets_to_anchors(self, anchor, target):
        from utils3d.geometric_torch import angle_dif
        if target.bbox3d.shape[0] == 0:
          matched_idxs = torch.ones([anchor.bbox3d.shape[0]], dtype=torch.int64, device=anchor.bbox3d.device) * (-1)
          matched_targets = anchor
        else:
          match_quality_matrix = boxlist_iou_3d(target, anchor, aug_thickness = self.aug_thickness, criterion=self.IoU_criteria, flag='rpn_label_generation')
          yaw_diff = angle_dif(anchor.bbox3d[:,-1].view(1,-1),  target.bbox3d[:,-1].view(-1,1), 0)
          yaw_diff = torch.abs(yaw_diff)
          cendis = anchor.bbox3d[:,0:3].view(1,-1,3) - target.bbox3d[:,0:3].view(-1,1,3)
          cendis = cendis.norm(dim=2)
          matched_idxs = self.proposal_matcher(match_quality_matrix, yaw_diff=yaw_diff, flag='RPN', cendis=cendis)
          #anchor.show__together(target, 200)
          # RPN doesn't need any fields from target
          # for creating the labels, so clear them all
          target = target.copy()
          # get the targets corresponding GT for each anchor
          # NB: need to clamp the indices because we can have a single
          # GT in the image, and matched_idxs can be -2, which goes
          # out of bounds
          matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        if SHOW_POS_NEG_ANCHORS:
            self.matched_idxs = matched_idxs
            self.matched_iou = match_quality_matrix.max(0)[0]

        if SHOW_IGNORED_ANCHOR:
          sampled_ign_inds = torch.nonzero(matched_idxs==-2).squeeze(1)
          anchors_ign = anchor[sampled_ign_inds]
          ignored_iou = match_quality_matrix.max(0)[0][sampled_ign_inds]
          print(f'\n ignore {len(anchors_ign)} anchors')
          print(f'\n ignore ious:\n{ignored_iou}')
          anchors_ign.show__together(target)
          #for i in range(0, len(anchors_ign), 3):
          #  ids_ign = [j for j in range(i, i+3)]
          #  print(f'iou: {ignored_iou[ids_ign]}')
          #  anchors_ign[ids_ign].show__together(target)
          pass

        if CHECK_MATCHER and target.bbox3d.shape[0]>0:
          check_matcher(target, anchor, match_quality_matrix, matched_idxs, self.dset_metas)

        if SHOW_POS_ANCHOR_IOU_SAME_LOC:
          num_gt = target.bbox3d.shape[0]
          for j in range(num_gt):
            sampled_pos_inds = torch.nonzero(matched_idxs==j).squeeze(1)
            inds_same_loc = anchor.same_loc_anchors(sampled_pos_inds)
            matched_idxs_same_loc = matched_idxs[inds_same_loc]

            # all the ious for gt box j
            iou_j = match_quality_matrix[j][sampled_pos_inds]
            anchors_pos_j = anchor[sampled_pos_inds]
            print(f'\n{iou_j.shape[0]} anchor matched as positive. All anchor centroids are shown.')
            print(f'ious:{iou_j}\n')
            anchors_pos_j.show__together(target[j], points=anchor.bbox3d[:,0:3])

            for i in range(iou_j.shape[0]):

              print(f'\n{i}th pos anchor for gt box j\n iou: {iou_j[i]}')
              #anchors_pos_j[i].show__together(target[j])

              ious_same_loc = match_quality_matrix[j][inds_same_loc[i]]
              yaw_diff_same_loc = yaw_diff[j,inds_same_loc[i]]
              print(f'\nall same loc anchors \nious:{ious_same_loc}\nmatched_idxs:{matched_idxs_same_loc[i]}\nyaw_diff:{yaw_diff_same_loc}')
              print(f'-1:low, -2:between')
              anchors_same_loc_i = anchor[inds_same_loc[i]]
              anchors_same_loc_i.show__together(target[j])
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            pass

        return matched_targets

    def prepare_targets(self, anchors, targets):
        '''
        labels: batch_size * []
        '''
        labels = []
        regression_targets = []
        batch_size = anchors.batch_size()
        assert batch_size == len(targets)
        for bi in range(batch_size):
            # merge anchors of all scales
            anchors_per_image = anchors.example(bi)
            targets_per_image = targets[bi]

            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            #labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox3d, anchors_per_image.bbox3d
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)


        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets, debugs=None):
        """
        Arguments:
            anchors (BoxList): box num: N
            objectness (list[Tensor]): len=scale_num
            box_regression (list[Tensor]): len=scale_num
            targets (list[BoxList]): len = batch size

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds0, sampled_neg_inds0 = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds0, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds0, dim=0)).squeeze(1)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        batch_size = anchors.batch_size()

        if SHOW_POS_NEG_ANCHORS:
          self.show_pos_neg_anchors(anchors, sampled_pos_inds, sampled_neg_inds, targets)

        if SHOW_PRED_POS_ANCHORS:
            self.show_pos_anchors_pred(box_regression, anchors, objectness, targets, sampled_pos_inds, sampled_neg_inds, regression_targets)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        if CHECK_REGRESSION_TARGET_YAW:
            rpn_target_yaw = regression_targets[sampled_pos_inds]
            print(f'max_rpn_target_yaw: {rpn_target_yaw.max()}')
            print(f'min_rpn_target_yaw: {rpn_target_yaw.min()}')
            assert rpn_target_yaw.max() < 1.5
            assert rpn_target_yaw.min() > -1.5

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            anchors[sampled_pos_inds].bbox3d,
            beta=1.0 / 9,
            size_average=False,
            yaw_loss_mode = self.yaw_loss_mode,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def show_pos_neg_anchors(self, anchors, sampled_pos_inds, sampled_neg_inds, targets):
      show_pos_anchors_for_each_label = True
      labels_all = list(self.dset_metas.label_2_class.keys())
      labels_all = [l for  l in labels_all if l!=0]
      assert anchors.batch_size()  == 1
      anchors.add_field('matched_idxs', self.matched_idxs.cpu().data.numpy().astype(np.int))
      anchors.add_field('matched_ious', self.matched_iou.cpu().data.numpy().astype(np.float))
      pos_inds_examples = anchors.seperate_items_to_examples(sampled_pos_inds)
      neg_inds_examples = anchors.seperate_items_to_examples(sampled_neg_inds)
      bs = anchors.batch_size()
      for bi in range(bs):
        targets_bi = targets[bi]
        labels_bi = targets_bi.get_field('labels').cpu().data.numpy()
        #label_nums = [ np.sum(labels_bi==l) for l in labels_all ]
        anchors_bi = anchors.example(bi)
        pos_anchors_bi = anchors_bi[pos_inds_examples[bi]]
        neg_anchors_bi = anchors_bi[neg_inds_examples[bi]]
        matched_idxs = pos_anchors_bi.get_field('matched_idxs').cpu().data.numpy().astype(np.int)
        matched_ious = pos_anchors_bi.get_field('matched_ious').cpu().data.numpy().astype(np.float)
        neg_ious = neg_anchors_bi.get_field('matched_ious').cpu().data.numpy().astype(np.float)
        labels_pos = labels_bi[matched_idxs]
        #pos_label_nums = [ np.sum(labels_pos==l) for l in labels_all ]
        mask_targ = np.ones(len(targets_bi))
        mask_targ[matched_idxs] = 0
        missed_targets_ids = np.nonzero(mask_targ)[0]
        missed_targets_label = labels_bi[missed_targets_ids]
        missed_targets_name = [self.dset_metas.label_2_class[l] for l in missed_targets_label]
        missed_targets = targets_bi[missed_targets_ids]

        print(f'{len(targets_bi)} targets')
        print(f'\n{len(targets[bi])} positive anchors')
        print(f'labels_pos: \n {labels_pos}')
        mean_matched_iou = matched_ious.mean()
        print(f'matched_ious:{matched_ious}\nmean_matched_iou:{mean_matched_iou}')
        print(f'missed_targets_ids: {missed_targets_ids}')
        print(f'missed_targets_label: {missed_targets_label}, {missed_targets_name}')
        if len(missed_targets_ids)>0:
            print('missed targets')
            targets_bi.show_highlight(missed_targets_ids)
        else:
            print('no target missed')
        pos_anchors_bi.show__together(targets[bi])
        low_pos_ids = np.argsort(matched_ious)[0:5]
        print(f'low pos ious: {matched_ious[low_pos_ids]}')
        pos_anchors_bi[low_pos_ids].show__together(targets[bi])
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

        biou_ids = np.argsort(neg_ious)[-5:]
        print(f'neg ious: {neg_ious[biou_ids]}')
        neg_anchors_bi[biou_ids].show__together(targets[bi])


        if show_pos_anchors_for_each_label:
          for l in labels_all:
              mask_l = labels_pos == l
              idxs_l = np.nonzero(mask_l)[0]
              class_name = self.dset_metas.label_2_class[ l ]
              print(f'pos anchors for {class_name}')
              if len(idxs_l)==0:
                  print('no pos anchors')
              else:
                  pos_anchors_bi[idxs_l].show__together(targets_bi)

      pass

    def show_pos_anchors_pred(self, rpn_box_regression, anchors, objectness,
                targets, sampled_pos_inds, sampled_neg_inds, regression_targets):
        pred_boxes_3d = self.box_coder.decode(rpn_box_regression, anchors.bbox3d)
        objectness_normed = objectness.sigmoid()
        pred_boxes = anchors.copy()
        pred_boxes.bbox3d = pred_boxes_3d
        pred_boxes.add_field('objectness', objectness_normed)

        pred_boxes.constants['type'] = 'prediction'
        sampled_pos_inds = pred_boxes.seperate_items_to_examples(sampled_pos_inds)
        sampled_neg_inds = pred_boxes.seperate_items_to_examples(sampled_neg_inds)

        for bi,predb in enumerate(pred_boxes.seperate_examples()):
          print('the top predicted objectness')
          predb.show_by_objectness(0.9, targets[bi], rpn_box_regression, anchors, regression_targets)
          print('all the predictions of positive anchors')
          predb.show_by_pos_anchor(sampled_pos_inds[bi], sampled_neg_inds[bi], targets[bi])
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
        yaw_threshold = cfg.MODEL.RPN.YAW_THRESHOLD
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )
    ay = cfg.MODEL.RPN.LABEL_AUG_THICKNESS_Y_TAR_ANC
    az = cfg.MODEL.RPN.LABEL_AUG_THICKNESS_Z_TAR_ANC
    aug_thickness = {'target_Y':ay[0], 'anchor_Y':ay[1],'target_Z':az[0], 'anchor_Z':az[1], }
    dset_metas = DSET_METAS(cfg.INPUT.CLASSES)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder, cfg.MODEL.LOSS.YAW_MODE, aug_thickness, dset_metas=dset_metas, IoU_criteria=cfg.MODEL.IOU_CRITERIA)
    return loss_evaluator

