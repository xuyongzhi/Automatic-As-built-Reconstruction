# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_iou_3d
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box_3d import cat_boxlist_3d
from maskrcnn_benchmark.modeling.seperate_classifier import SeperateClassifier
from maskrcnn_benchmark.structures.bounding_box_3d import extract_order_ids

DEBUG = True
SHOW_ROI_CLASSFICATION = DEBUG and 0
CHECK_IOU = False
CHECK_REGRESSION_TARGET_YAW = False

def get_prop_ids_per_targ(matched_idxs, tg_corner_connect_ids, proposals=None, targets=None):
  '''
  matched_idxs: [n] n=number of proposal, the matched target id for each proposal
      -1: not matched, >=0: target id
  tg_corner_connect_ids: [2t,3]: t=target number, 2t=corner number. the 3 connected corners of each corner. If a corner is not connected, asign -1.

  connect_proC_ids_each_proC: [2p,6] p=number of positive proposal, 2p=number of corners of positive proposals
                    the corner ids of 6 connected corners among all corners of positive proposals
  pos_prop_ids: [p] the ids of positive proposals, sorted by the same order with connect_proC_ids_each_proC
  '''
  debug = proposals is not None and 0

  t = tg_corner_connect_ids.shape[0] // 2
  pro_num = matched_idxs.shape[0]
  device = matched_idxs.device
  tccn = tg_corner_connect_ids.shape[1] # the maximum connected corners of target: 3
  assert tccn == 3
  etpn = 4 # the maximum detected proposal number for each target
  # the positive proposal ids

  # Add self to connected ids: solve the porblem that multiple proposals for one
  # target, these proposals should be connected
  tmp = torch.arange(t*2, device=device, dtype=torch.int64).view(t*2,1)
  tg_corner_connect_ids = torch.cat([tmp, tg_corner_connect_ids], 1)

  pos_prop_ids = torch.nonzero(matched_idxs >= 0).squeeze(dim=1) # [p]
  p = pos_prop_ids.shape[0]
  # the matched target ids of postive proposals
  matched_tar_idxs_pos = matched_idxs[pos_prop_ids].squeeze() # [p]

  # get the pos_proposal ids for each target
  tar_ids_each_pro, sorting = matched_tar_idxs_pos.sort()
  tar_ids_each_pro = tar_ids_each_pro.view([-1])
  pos_prop_ids = pos_prop_ids[sorting] # [p]

  # tarC_ids_each_proC: the responding target corner index of each positive
  # proposal corner
  tmp = tar_ids_each_pro.view(-1,1).repeat(1,2)*2
  tmp[:,1] += 1
  tarC_ids_each_proC = tmp.view(-1) # [2p]

  if debug:
    check_corner_responding_tp(tar_ids_each_pro, tarC_ids_each_proC, proposals[pos_prop_ids], targets)

  # connect_tarC_ids_each_proC: the responing connected target corner ids for
  # each proposal corner
  connect_tarC_ids_each_proC = tg_corner_connect_ids[tarC_ids_each_proC] # [2p,3]

  # proC_ids_each_tarC: proposal corner index of each target corner
  pro_ids_each_tar = torch.zeros([t, etpn], dtype=torch.int64, device=device) -1# [2t,4]
  for j in range(etpn):
    ids_jth = extract_order_ids(tar_ids_each_pro, j)
    if ids_jth.shape[0]  > 0:
      tar_ids_j = tar_ids_each_pro[ids_jth]
      pro_ids_each_tar[tar_ids_j, j] = ids_jth

  proC_ids_each_tarC = pro_ids_each_tar.unsqueeze(1).repeat(1,2,1) * 2
  proC_ids_each_tarC[:,1,:] += 1
  proC_ids_each_tarC = proC_ids_each_tarC.view([2*t, etpn])

  # for each proC, the connected proC ids
  connect_proC_ids_each_proC = torch.zeros([2*p,etpn*tccn], dtype=torch.int64, device=device) -1 # [2p,4]
  tmp0 = proC_ids_each_tarC[0:1]*0 -100
  tmp = torch.cat([tmp0, proC_ids_each_tarC ], 0)
  connect_proC_ids_each_proC = tmp[ connect_tarC_ids_each_proC + 1 ].view([2*p,-1])
  connect_proC_ids_each_proC,_ = (-connect_proC_ids_each_proC).sort(dim=1)
  connect_proC_ids_each_proC = -connect_proC_ids_each_proC
  connect_proC_ids_each_proC = connect_proC_ids_each_proC[:,0:8]


  # <0 to -1
  mask = connect_proC_ids_each_proC < 0
  connect_proC_ids_each_proC[mask] = -1
  if debug:
    #check_grouping_labels(targets, tg_corner_connect_ids)
    check_grouping_labels(proposals[pos_prop_ids], connect_proC_ids_each_proC)

  return connect_proC_ids_each_proC,  pos_prop_ids


def check_corner_responding_tp(tar_ids_each_pro, tarC_ids_each_proC, proposals, targets):
  pro_box = proposals.bbox3d
  the_pro_box = targets.bbox3d[tar_ids_each_pro]


  proCorners, _ = proposals.get_2top_corners_offseted()
  tarCorners, _ = targets.get_2top_corners_offseted()
  proCorners = proCorners.view(-1,3)
  tarCorners = tarCorners.view(-1,3)
  the_tarCorners = tarCorners[tarC_ids_each_proC]
  dif = torch.abs(proCorners - the_tarCorners)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def check_grouping_labels( proposals, connect_proC_ids_each_proC):
      assert len(proposals)*2 == connect_proC_ids_each_proC.shape[0]
      corners, _ = proposals.get_2top_corners_offseted()
      corners = corners.view([-1,3])
      n = corners.shape[0]
      for i in range( min(n,10) ):
        ids_i = connect_proC_ids_each_proC[i]
        mask  = ids_i>=0
        ids_i = ids_i[mask].view([-1])
        if ids_i.shape[0]>0:
          tmp = ids_i[0:1]*0 + i
          ids_i = torch.cat([tmp,  ids_i], 0)
          corners0 = corners[ids_i]
          print(f'corners:\n{corners0}')
          proposals.show(points = corners0)
          pass
      pass

def check_corner_semantics(cor_xyzs):
  pass

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, yaw_loss_mode, add_gt_proposals, aug_thickness, seperate_classifier, class_specific):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder3D)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.yaw_loss_mode = yaw_loss_mode

        self.high_threshold = proposal_matcher.high_threshold
        self.low_threshold = proposal_matcher.low_threshold
        self.add_gt_proposals = add_gt_proposals
        self.aug_thickness = aug_thickness
        self.seperate_classifier = seperate_classifier
        self.need_seperate = seperate_classifier.need_seperate
        self.class_specific = class_specific

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou_3d(target, proposal, aug_thickness=self.aug_thickness, criterion=-1, flag='roi_label_generation')
        matched_idxs = self.proposal_matcher(match_quality_matrix, yaw_diff=None, flag='ROI')
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        if CHECK_IOU:
          num_gt = len(target)
          if not torch.all( matched_idxs[-num_gt:].cpu() == torch.arange(num_gt) ):
            ious = match_quality_matrix[:,-num_gt:].diag()
            err_inds = torch.nonzero(torch.abs(ious - 1) > 1e-5 ).view(-1) - len(ious)
            print( f"IOU error: \n{ious}")
            err_targets = target[err_inds]
            ious__ = boxlist_iou_3d(err_targets, err_targets, 0)
            print(err_targets.bbox3d)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            assert False
            pass
        return matched_targets

    def prepare_targets(self, proposals, targets):
        '''
        proposals do not have object class info
        ROI is only performed on matched proposals.
        Generate class label and regression_targets for all matched proposals.
        '''
        labels = []
        regression_targets = []
        connect_proC_ids_each_proCs = []
        pos_prop_ids = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if len(targets_per_image) == 0:
              prop_num = len(proposals_per_image)
              # negative
              device = proposals[0].bbox3d.device
              labels.append(torch.zeros([prop_num],dtype=torch.int64).to(device))
              regression_targets.append(torch.zeros([prop_num,7],dtype=torch.float32).to(device))
              continue

            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image0 = matched_targets.get_field("labels")
            labels_per_image = labels_per_image0.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox3d, proposals_per_image.bbox3d
            )

            # grouping
            tg_corner_connect_ids = targets_per_image.get_connect_corner_ids()
            connect_proC_ids_each_proC_, pos_prop_ids_ = get_prop_ids_per_targ(matched_idxs, tg_corner_connect_ids, proposals_per_image, targets_per_image)

            connect_proC_ids_each_proCs.append(connect_proC_ids_each_proC_)
            pos_prop_ids.append(pos_prop_ids_)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets, connect_proC_ids_each_proCs, pos_prop_ids


    def subsample(self, proposals, targets):
      if self.need_seperate:
        proposals = self.seperate_classifier.seperate_subsample(proposals, targets, self.subsample_standard)
        self._proposals = proposals
        return proposals
      else:
        return self.subsample_standard(proposals, targets)


    def subsample_standard(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, regression_targets, connect_proC_ids_each_proCs, pos_prop_ids = self.prepare_targets(proposals, targets)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            if labels_per_image.shape[0] != proposals_per_image.bbox3d.shape[0]:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("regression_targets", regression_targets_per_image )

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # rm ignored proposals
        for img_idx, (pos_inds_img, neg_inds_img, pos_prop_ids_per_image) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds, pos_prop_ids)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

            # mapping raw index among all proposals to new which removed ignored
            n = pos_inds_img.shape[0]
            tmp = torch.zeros(n, dtype=torch.int64, device=img_sampled_inds.device) - 1
            m = img_sampled_inds.shape[0]
            tmp[img_sampled_inds] = torch.arange(m, dtype=torch.int64, device=img_sampled_inds.device)
            pos_prop_ids[img_idx] =  tmp[pos_prop_ids_per_image]

        self._proposals = proposals
        self._connect_proC_ids_each_proC = connect_proC_ids_each_proCs
        self._pos_prop_ids  = pos_prop_ids
        return proposals

    def __call__(self, class_logits, box_regression, corners_semantic, targets=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits: [n,class_num]
            box_regression: class agnostic [n,7]; class specific: [n,7*num_classes]
            corners_semantic: [n,8*2]
            targets for debuging only: [BoxList3D] len = batch_size

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        #class_logits = cat(class_logits, dim=0)
        n = class_logits.shape[0]
        assert n == box_regression.shape[0]
        assert  corners_semantic is None or n == corners_semantic.shape[0]
        class_num = class_logits.shape[1]

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        proposals = cat_boxlist_3d(proposals, per_example=True)
        labels = proposals.get_field("labels")
        regression_targets = proposals.get_field("regression_targets")
        pro_bbox3ds = proposals.bbox3d

        if not self.need_seperate:
          classification_loss = F.cross_entropy(class_logits, labels)
          box_loss, corner_loss = self.box_loss(labels, box_regression, regression_targets, pro_bbox3ds, corners_semantic)
        else:
          classification_loss = self.seperate_classifier.roi_cross_entropy_seperated(class_logits, labels, proposals)
          box_loss, corner_loss = self.seperate_classifier.roi_box_loss_seperated(self.box_loss,
                                  labels, box_regression, regression_targets,
                                  pro_bbox3ds = pro_bbox3ds, corners_semantic = corners_semantic )
          pass

        if SHOW_ROI_CLASSFICATION:
          self.show_roi_cls_regs(proposals, classification_loss, box_loss, class_logits,  targets, box_regression, regression_targets)

        return classification_loss, box_loss, corner_loss


    def box_loss(self, labels, box_regression, regression_targets, bbox3ds, corners_semantic):
        '''
        class_specific:
          labels:[n], box_regression:[n,c*7], regression_targets:[n,7], bbox3ds:[n,7]
        '''
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        device = box_regression.device
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.class_specific:
          map_inds = 7 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device)
          box_regression_pos = box_regression[sampled_pos_inds_subset[:, None], map_inds]
        else:
          box_regression_pos = box_regression[sampled_pos_inds_subset, :]

        regression_targets_pos = regression_targets[sampled_pos_inds_subset]

        if CHECK_REGRESSION_TARGET_YAW:
            roi_target_yaw = regression_targets_pos[:,-1]
            print(f'max_roi_target_yaw: {roi_target_yaw.max()}')
            print(f'min_roi_target_yaw: {roi_target_yaw.min()}')
            assert roi_target_yaw.max() < 1.5
            assert roi_target_yaw.min() > -1.5

        box_loss = smooth_l1_loss(
            box_regression_pos,
            regression_targets_pos,
            bbox3ds[sampled_pos_inds_subset],
            size_average=False,
            beta=1 / 5.,  # 1
            yaw_loss_mode = self.yaw_loss_mode
        )
        box_loss = box_loss / labels.numel()
        if corners_semantic is None:
          corner_loss = {}
        else:
          corner_loss = self.corner_connection_loss(corners_semantic)
        return box_loss, corner_loss

    def corner_connection_loss(self, corners_semantic, active_threshold=0.2):
        '''
        corners_semantic: [batch_size * n, sem_c]
        active_threshold: only corner distaces within this threshold are calculated
        '''
        def cor_geo_pull_loss_f(cor_ids, corners, flag):
            assert flag == 'geo' or flag == 'sem'
            assert cor_ids.shape[0] == corners.shape[0]

            device = cor_ids.device
            n = cor_ids.shape[0]

            tmp = corners[0:1] * 0 + float('NaN')
            corners = torch.cat([ tmp, corners ], 0)

            connect_corners = corners[cor_ids+1]
            c = corners.shape[1]
            corners = corners[1:].view(-1,1,c)
            dif = connect_corners - corners
            dif = dif.view(-1,c)
            mask0 = ~torch.isnan(dif[:,0])
            dif_valid = dif[mask0]
            dis_valid = dif_valid.norm(dim=1)
            pull_loss_i  = dis_valid

            mask_valid = pull_loss_i < active_threshold
            pull_loss_i = pull_loss_i[mask_valid]

            one = torch.ones(1, dtype=torch.int64, device=device).squeeze()
            pull_num = torch.max( mask_valid.sum() - n, one )
            pull_loss_i = pull_loss_i.sum() / pull_num
            return pull_loss_i

        def cor_semantic_loss_f(cor_ids, semantic, cor_xyzs, delta=1.0, geo_close_threshold=0.5):
          n = cor_ids.shape[0]
          m = cor_ids.shape[1]
          c = semantic.shape[1]
          device = cor_ids.device

          sem_dif = semantic.view(1,n,c) - semantic.view(n,1,c)
          sem_dis = sem_dif.norm(dim=2)

          # not connect: 1,  connect: 0
          mask_not_connect = torch.ones(n, n, dtype=torch.int32, device=cor_ids.device)
          cor_ids_0 = torch.arange(n).view(n,1).to(device)
          cor_ids1 = torch.cat([cor_ids_0, cor_ids], 1)
          cor_ids1 = cor_ids1.contiguous().view([-1,1])
          mask_valid = cor_ids1 >=0
          cor_ids1 = cor_ids1[mask_valid]
          cor_ids2 = cor_ids_0.repeat(1,m+1).view([-1,1])[mask_valid]
          mask_not_connect[cor_ids1, cor_ids2] = 0
          mask_connect = 1-mask_not_connect

          # geometric close mask
          geo_dif = cor_xyzs.view(n,1,3) - cor_xyzs.view(1,n,3)
          geo_dis = geo_dif.norm(dim=2)
          close_mask = (geo_dis < geo_close_threshold).to(torch.int32)

          # pull loss: connected corners
          sem_dis_pull = sem_dis  * mask_connect.to(torch.float32)
          one = torch.ones(1, dtype=torch.int64, device=device).squeeze()
          pull_num = torch.max( mask_connect.sum() - n, one )
          sem_pull_loss = sem_dis_pull.sum() / pull_num

          # push loss: not connected & close corners
          push_mask = mask_not_connect * close_mask
          sem_dis_push0 = (delta - sem_dis)  * push_mask.to(torch.float32)
          sem_push_loss = torch.clamp( sem_dis_push0, min=0)
          push_num = torch.max( push_mask.sum(), one )
          sem_push_loss = sem_push_loss.sum() / push_num

          #Bbox3D.draw_points(cor_xyzs[[0,59, 57, 55, 53]].cpu().data.numpy())
          return sem_push_loss, sem_pull_loss


        batch_size = len(self._proposals)
        assert batch_size == len(self._pos_prop_ids)
        if len(self._pos_prop_ids) == 0:
          zero = torch.zeros(1, dtype=torch.float32, device = corners_semantic.device).squeeze()
          corner_loss = { 'geometric_pull_loss':  zero,
                        'semantic_pull_loss':     zero,
                        'semantic_push_loss':     zero }
          return corner_loss

        sem_c = corners_semantic.shape[1]
        corners_semantic = corners_semantic.view(batch_size, -1, sem_c)

        geometric_pull_loss = []
        semantic_pull_loss = []
        semantic_push_loss = []
        for i in range(batch_size):
            cor_xyzs0, _ = self._proposals[i][self._pos_prop_ids[i]].get_2top_corners_offseted()
            cor_xyzs = cor_xyzs0.view(-1,3)

            semantic = corners_semantic[i][self._pos_prop_ids[i]]
            c = semantic.shape[1]
            semantic = semantic.view(-1,c//2)

            cor_ids = self._connect_proC_ids_each_proC[i]

            #check_grouping_labels( self._proposals[i][self._pos_prop_ids[i]], cor_ids )

            geo_pull_loss_i = cor_geo_pull_loss_f(cor_ids, cor_xyzs, 'geo')
            sem_push_loss_i, sem_pull_loss_i = cor_semantic_loss_f(cor_ids, semantic, cor_xyzs)

            geometric_pull_loss.append(geo_pull_loss_i)
            semantic_pull_loss.append(sem_pull_loss_i)
            semantic_push_loss.append(sem_push_loss_i)

        geometric_pull_loss = sum(geometric_pull_loss) / batch_size
        semantic_pull_loss = sum(semantic_pull_loss) / batch_size
        semantic_push_loss = sum(semantic_push_loss) / batch_size
        corner_loss = { 'geometric_pull_loss': geometric_pull_loss,
                       'semantic_pull_loss':  semantic_pull_loss,
                       'semantic_push_loss':  semantic_push_loss  }
        return corner_loss


    def show_roi_cls_regs(self, proposals, classification_loss, box_loss,
              class_logits, targets,  box_regression, regression_targets):
          '''
          From rpn nms: FP, FN, TP
          ROI: (1)remove all FP (2) add all FN, (3) keep all TP
          '''
          assert proposals.batch_size() == 1
          targets = cat_boxlist_3d(targets, per_example=True)
          roi_class_pred = F.softmax(class_logits)
          pred_logits = torch.argmax(class_logits, 1)
          labels = proposals.get_field("labels")
          metric_inds, metric_evals = proposals.metric_4areas(self.low_threshold, self.high_threshold)
          gt_num = len(targets)
          device = class_logits.device
          num_classes = class_logits.shape[1]

          class_err = (labels != pred_logits).sum()

          print('\n-----------------------------------------\n roi classificatio\n')
          print(f"RPN_NMS: {metric_evals}")
          print(f"classification_loss:{classification_loss}, box_loss: {box_loss}")

          def show_one_type(eval_type):
              indices = metric_inds[eval_type]
              if eval_type == 'TP' and self.add_gt_proposals:
                indices = indices[0:-gt_num]
              n0 = indices.shape[0]
              pro_ = proposals[indices]
              objectness_ = pro_.get_field('objectness')
              logits_ = pred_logits[indices]
              labels_ = labels[indices]

              err_ = torch.abs(logits_ - labels_)
              err_num = err_.sum()
              print(f"\n * * * * * * * * \n{eval_type} :{n0} err num: {err_num}")
              print(f"objectness_:{objectness_}\n")
              if n0 > 0:
                roi_class_pred_ = roi_class_pred[indices[:,None], labels_[:,None]]

                #if eval_type != 'TP':
                print(f"roi_class_pred_:\n{roi_class_pred_}")

                if eval_type == 'FP':
                  pro_.show__together(targets)
                  pass

                if eval_type == 'FN' or eval_type == 'TP':
                  map_inds_ = 7 * labels_[:, None] + torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device)
                  roi_box_regression_ = box_regression[indices[:,None], map_inds_]
                  roi_box = self.box_coder.decode(roi_box_regression_, pro_.bbox3d)
                  tar_reg = regression_targets[indices]
                  #roi_box = self.box_coder.decode(tar_reg, pro_.bbox3d)
                  print(f"target reg: \n{tar_reg[0:3]}")
                  print(f"roi_reg: \n{roi_box_regression_[0:3]}")

                  roi_box[:,0] += 15
                  roi_boxlist_ = pro_.copy()
                  roi_boxlist_.bbox3d = roi_box

                  targets_ = targets.copy()
                  targets_.bbox3d[:,0] += 15

                  bs_ = cat_boxlist_3d([pro_, roi_boxlist_], per_example = False)
                  tg_ = cat_boxlist_3d([targets, targets_], False)
                  bs_.show__together(tg_, twolabels=True)

                  pass
              pass

          show_one_type('FP')
          show_one_type('FN')
          show_one_type('TP')
          return


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder3D(is_corner_roi = cfg.MODEL.CORNER_ROI,  weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )
    yaw_loss_mode = cfg.MODEL.LOSS.YAW_MODE
    add_gt_proposals = cfg.MODEL.RPN.ADD_GT_PROPOSALS
    ay = cfg.MODEL.ROI_HEADS.LABEL_AUG_THICKNESS_Y_TAR_ANC
    az = cfg.MODEL.ROI_HEADS.LABEL_AUG_THICKNESS_Z_TAR_ANC
    aug_thickness = {'target_Y':ay[0], 'anchor_Y':ay[1], 'target_Z':az[0], 'anchor_Z':az[1]}
    in_classes = cfg.INPUT.CLASSES
    num_input_classes = len(in_classes)

    seperate_classes = cfg.MODEL.SEPARATE_CLASSES_ID
    class_specific = cfg.MODEL.CLASS_SPECIFIC
    seperate_classifier = SeperateClassifier( seperate_classes, num_input_classes, class_specific, 'ROI_LOSS')

    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder,
                    yaw_loss_mode, add_gt_proposals, aug_thickness, seperate_classifier,
                                             class_specific=class_specific)

    return loss_evaluator, seperate_classifier

