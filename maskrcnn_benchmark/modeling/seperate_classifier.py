import torch
from torch.nn import functional as F
from maskrcnn_benchmark.structures.bounding_box_3d import cat_boxlist_3d
from data3d.dataset_metas  import DSET_METAS

DEBUG = False
WALL_ID = DSET_METAS.class_2_label0['wall']
SHOW_PROPOSALS = 0

class SeperateClassifier():
    def __init__(self, seperate_classes, num_input_classes, class_specific, flag ):
      '''
      (1) For RPN
      Each feature predict two proposals, one for seperated classes, the other one for remaining classes
      (2) For ROI
      Add a background label for the seperated classes at the end. As a result, the dimension of predicted classes increases by 1.
      The dimension of predicted boxes increases by 7.

      0: the seperated classes
      1: the remaining classes
      '''
      self.need_seperate = len(seperate_classes) > 0
      if not self.need_seperate:
        return

      self.flag = flag
      self.class_specific = class_specific

      [sp.sort() for sp in seperate_classes]
      self.num_input_classes = num_input_classes # include background
      sepcls_flat = [c for cs in seperate_classes for c in cs]
      assert 0 not in sepcls_flat
      remaining_classes = [[c for c in range(num_input_classes) if c not in sepcls_flat]]
      tmp = num_input_classes
      seperate_classes_intact = []
      for sep in seperate_classes:
        seperate_classes_intact.append( [tmp]+sep ) # background is the first
        tmp += 1
      self.grouped_classes = remaining_classes + seperate_classes_intact
      self.group_num = len(self.grouped_classes)
      self.seperated_num_classes_total = num_input_classes + self.group_num-1
      self.class_nums = [len(gc) for gc in self.grouped_classes]


      self.org_labels_to_sep_labels = torch.zeros([num_input_classes+self.group_num-1,2], dtype=torch.int32)-1
      self.sep_labels_to_org_labels = [torch.ones([n], dtype=torch.int32) * (-1) for n in self.class_nums]
      for g in range(self.group_num):
        for i, c in enumerate(self.grouped_classes[g]):
          self.org_labels_to_sep_labels[c] = torch.tensor([g,i]) # 0 not in seperate_classes
          self.sep_labels_to_org_labels[g][i] = c

      for i,gcls in enumerate(self.grouped_classes):
        if WALL_ID in gcls:
          self.group_id_include_wall = i

      if DEBUG:
        print(f'\n\nseperate_classes: {seperate_classes}')
        print(f'labels0_to_org_labels: {self.labels0_to_org_labels}')
        print(f'org_labels_to_labels0: {self.org_labels_to_labels0}')
        print(f'labels1_to_org_labels: {self.labels1_to_org_labels}')
        print(f'org_labels_to_labels1: {self.org_labels_to_labels1}')

      pass

    #---------------------------------------------------------------------------
    # For RPN
    #---------------------------------------------------------------------------
    def seperate_rpn_assin(self, targets):
      self.targets_groups_rpn = self.seperate_targets_and_update_labels(targets)

    def seperate_rpn_selector(self, box_selector_fn, anchors, objectness, rpn_box_regression, targets, add_gt_proposals):
      '''
        objectness: [n,2]
        rpn_box_regression: [n,14]
        targets: labels 0~nc_total
        self.targets0: labels 0~nc_0
        self.targets1: labels 0~nc_1
      '''
      assert objectness.shape[1] == self.group_num
      assert rpn_box_regression.shape[1] == self.group_num * 7
      self.targets_groups_rpn = self.seperate_targets_and_update_labels(targets)
      boxes_g = []
      for gi in range(self.group_num):
        boxes_i = box_selector_fn(anchors, objectness[:,gi], rpn_box_regression[:,gi*7:gi*7+7], self.targets_groups_rpn[gi], add_gt_proposals)
        boxes_g.append(boxes_i)


      if DEBUG and False:
        show_box_fields(targets, 'A')
        show_box_fields(self.targets0, 'B')
        show_box_fields(self.targets1, 'C')
        show_box_fields(boxes0, 'D')
        show_box_fields(boxes1, 'E')
        show_box_fields(anchors, 'F')
      return boxes_g

    def seperate_rpn_loss_evaluator(self, loss_evaluator_fn, anchors, objectness, rpn_box_regression, targets, debugs={}):
      #targets0, targets1 = self.seperate_targets(targets)
      loss_objectness = []
      loss_rpn_box_reg = []
      for gi in range(self.group_num):
        loss_objectness_i, loss_rpn_box_reg_i = loss_evaluator_fn(anchors, objectness[:,gi], rpn_box_regression[:,gi*7:gi*7+7], self.targets_groups_rpn[gi])
        loss_objectness.append(loss_objectness_i)
        loss_rpn_box_reg.append(loss_rpn_box_reg_i)

      if DEBUG and False:
        show_box_fields(self.targets0, 'B')
        show_box_fields(self.targets1, 'C')
      return loss_objectness, loss_rpn_box_reg

    #---------------------------------------------------------------------------
    # For Detector
    #---------------------------------------------------------------------------
    def sep_roi_heads( self, roi_heads_fn, roi_features, proposals, targets, points=None):
      if DEBUG and False:
        show_box_fields(proposals, 'A')
      proposals = self.cat_boxlist_3d_seperated(proposals)
      if DEBUG and False:
        show_box_fields(proposals, 'B')

      if SHOW_PROPOSALS:
        self.show_proposals(proposals, points)
      return roi_heads_fn(roi_features, proposals, targets)

    def show_proposals(self, proposals, points):
      points = points[1][:,0:6]
      proposals.show_by_objectness(0.1, points=points, points_keep_rate=0.9, points_sample_rate=0.4)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    #---------------------------------------------------------------------------
    # For ROI
    #---------------------------------------------------------------------------
    def seperate_subsample(self, proposals, targets, subsample_fn):
        proposals_g, _ = self.seperate_proposals(proposals)
        self.targets_g = self.seperate_targets_and_update_labels(targets)

        for gi in range(self.group_num):
          proposals_g[gi] = subsample_fn(proposals_g[gi], self.targets_g[gi])
          assert self.targets_g[gi][0].get_field('labels').max() < self.class_nums[gi]

        bs = len(proposals)
        proposals_out = []
        for i in range(bs):
          psi = [proposals_g[j][i] for j in range(self.group_num)]
          proposals_out.append( cat_boxlist_3d(psi, per_example=False) )

        #assert self.targets_1[0].get_field('labels').max() <= self.num_classes1 - 1


        if DEBUG and False:
          show_box_fields(proposals, 'In')
          show_box_fields(proposals_0, 'Sep0')
          show_box_fields(proposals_1, 'Sep1')
          show_box_fields(proposals_0_, 'subs0')
          show_box_fields(proposals_1_, 'subs1')
          show_box_fields(proposals_out, 'Out')

          show_box_fields(self.targets_0, 'T0')
          show_box_fields(self.targets_1, 'T1')
        return proposals_out

    def roi_cross_entropy_seperated(self, class_logits, labels, proposals):
      '''
      class_logits: [n, num_classes+1]
      labels: [n]
      self.seperate_classes: [num_classes0] (not include 0)

      In the (num_classes+1) dims of class_logits, the first (num_classes0+1) dims are for self.seperate_classes,
      the following (num_classes1+1) are for the remianing.
      '''
      proposals_g, self.sep_ids_g_roi  = self.seperate_proposals(proposals)
      class_logits_g = self.seperate_pred_logits(class_logits, self.sep_ids_g_roi)
      self.labels_g_roi = []
      losses_g = []
      for gi in  range(self.group_num):
        self.labels_g_roi.append( labels[self.sep_ids_g_roi[gi]] )
        assert self.labels_g_roi[-1].max() <= self.class_nums[gi] - 1
        loss_i = F.cross_entropy(class_logits_g[gi], self.labels_g_roi[gi])
        losses_g.append(loss_i)

      return losses_g

    def roi_box_loss_seperated(self, box_loss_fn, labels, box_regression, regression_targets, pro_bbox3ds, corners_semantic):
        '''
        labels: [n,2]
        box_regression: [b,7*seperated_num_classes_total]
        regression_targets:[n,7,2]
        pro_bbox3ds:[n,7]
        '''
        box_regression_g, corners_semantic_g  = self.seperate_pred_box(box_regression, corners_semantic, self.sep_ids_g_roi)
        regression_targets_g = []
        pro_bbox3ds_g = []
        box_losses_g = []
        corner_losses_g = []
        for gi in  range(self.group_num):
            regression_targets_g.append(regression_targets[self.sep_ids_g_roi[gi]])
            pro_bbox3ds_g.append( pro_bbox3ds[self.sep_ids_g_roi[gi]] )
            box_loss, corner_loss =  box_loss_fn(self.labels_g_roi[gi], box_regression_g[gi], regression_targets_g[gi], pro_bbox3ds_g[gi], corners_semantic_g[gi])
            box_losses_g.append( box_loss )
            corner_losses_g.append(corner_loss)

        corner_losses_g = [corner_losses_g[self.group_id_include_wall]]
        return box_losses_g, corner_losses_g

    #---------------------------------------------------------------------------
    # Functions Utils
    #---------------------------------------------------------------------------
    def cat_boxlist_3d_seperated(self, bboxes_ls):
        batch_size = bboxes_ls[0].batch_size()
        m = len(bboxes_ls)
        assert m==self.group_num

        bboxes_ = [None]*m
        for gi in range(m):
          bboxes_ls[gi].add_field('sep_id', torch.ones([len(bboxes_ls[gi])], dtype=torch.int32)*gi )
          bboxes_[gi] = bboxes_ls[gi].seperate_examples()

        bboxes_ls_new = []
        for j in range(batch_size):
          bboxes_ls_new.append( cat_boxlist_3d([bboxes_[i][j] for i in range(m)], per_example=False) )
        bboxes_ls_new_all = cat_boxlist_3d(bboxes_ls_new, per_example=True)
        return bboxes_ls_new_all

    def seperate_proposals(self, proposals):
      bs = len(proposals)
      proposals_g = [[None for i in range(bs)] for j in range(self.group_num)]
      sep_ids_g = [[None for i in range(bs)] for j in range(self.group_num)]
      ids_cum_sum = 0

      for i in range(bs):
        sep_id = proposals[i].get_field('sep_id')

        for gi in range(self.group_num):
          sep_ids_gi = torch.nonzero(sep_id==gi).view([-1])
          sep_ids_g[gi][i] =  sep_ids_gi + ids_cum_sum
          proposals_g[gi][i] =  proposals[i][sep_ids_gi]

        ids_cum_sum += len(proposals[i])

      for gi in range(self.group_num):
        sep_ids_g[gi] = torch.cat(sep_ids_g[gi], 0)
      return proposals_g, sep_ids_g

    def seperate_pred_logits(self, class_logits, sep_ids_g):
      assert class_logits.shape[1] == self.seperated_num_classes_total
      assert class_logits.shape[0] == sum([sep.shape[0] for sep in sep_ids_g])
      class_logits_g = []
      for i in range(self.group_num):
        class_logits_i = class_logits[sep_ids_g[i],:] [:,self.grouped_classes[i]]
        class_logits_g.append(class_logits_i)
      return class_logits_g

    def seperate_pred_box(self, box_regression, corners_semantic, sep_ids_g):
      if self.class_specific:
        assert box_regression.shape[1] == self.seperated_num_classes_total*7
      else:
        assert box_regression.shape[1] == 7
      assert box_regression.shape[0] == sum([s.shape[0] for s in sep_ids_g])
      n = box_regression.shape[0]
      box_regression_g = []
      corners_semantic_g = []
      for i in range(self.group_num):
        if self.class_specific:
          box_regression_i = box_regression.view([n,-1,7])[:, self.grouped_classes[i], :].view([n,-1])[sep_ids_g[i]]
          print('corners_semantic is not implemented')
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        else:
          box_regression_i = box_regression[sep_ids_g[i]]
          if corners_semantic is None:
              corners_semantic_i = None
          else:
              corners_semantic_i = corners_semantic[sep_ids_g[i]]
        box_regression_g.append(box_regression_i)
        corners_semantic_g.append(corners_semantic_i)
      return box_regression_g, corners_semantic_g

    def _seperating_ids(self, labels):
      assert isinstance(labels, torch.Tensor)
      ids_groups = []
      for gc in self.grouped_classes:
          ids_g = []
          for c in gc:
              ids_c = torch.nonzero(labels==c).view([-1])
              ids_g.append(ids_c)
          ids_g = torch.cat(ids_g, 0)
          ids_groups.append(ids_g)
      return ids_groups

    def seperate_targets_and_update_labels(self, targets):
      targets_g = self.seperate_targets(targets)
      bs = len(targets)
      for bi in range(bs):
        for gi in range(len(targets_g)):
          org_labels_gi = targets_g[gi][bi].get_field('labels')
          sep_labels_gi = self.update_labels_to_seperated_id([org_labels_gi])
          targets_g[gi][bi].extra_fields['labels'] = sep_labels_gi[0]

      return targets_g

    def seperate_targets(self, targets):
        assert isinstance(targets, list)
        batch_size = len(targets)
        targets_groups = [[None]*batch_size for i in range(self.group_num)]
        for bi in range(batch_size):
          labels = targets[bi].get_field('labels')
          ids_groups = self._seperating_ids(labels)
          for gi in range(self.group_num):
            targets_groups[gi][bi] = targets[bi][ids_groups[gi]]
        return targets_groups

    def update_labels_to_seperated_id(self, labels_seperated_org):
      '''
      labels_seperated_org: the value is originally value of not seperated, but only part.
      '''
      assert isinstance(labels_seperated_org, list)
      labels_new = []
      device = labels_seperated_org[0].device
      org_to_new = self.org_labels_to_sep_labels
      for ls in labels_seperated_org:
        gid_lid = org_to_new[ls.long()].to(device).long()
        labels_new.append( gid_lid[:,1] )
        if labels_new[-1].shape[0] > 0:
          try:
            assert labels_new[-1].min() >= 0
          except:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
      return labels_new

    def turn_labels_back_to_org(self, result, sep_flag):
      if sep_flag == 0:
        l2ol = self.labels0_to_org_labels
      elif sep_flag == 1:
        l2ol = self.labels1_to_org_labels
      bs = len(result)
      for b in range(bs):
        result[b].extra_fields['labels'] =  l2ol[result[b].extra_fields['labels']]
      return result

    def post_processor(self, class_logits, box_regression, corners_semantic, proposals, post_processor_fn):
      proposals_g, sep_ids_g  = self.seperate_proposals(proposals)
      #for gi in range(self.group_num):
      class_logits_g = self.seperate_pred_logits(class_logits, sep_ids_g)
      box_regression_g, corners_semantic_g = self.seperate_pred_box(box_regression, corners_semantic, sep_ids_g)

      results_g = []
      for gi in range(self.group_num):
        result_gi = post_processor_fn( (class_logits_g[gi], box_regression_g[gi] , corners_semantic_g[gi] ), proposals_g[gi] )
        results_g.append(result_gi)

      batch_size = len(proposals)
      result = []
      for b in range(batch_size):
        for gi in range(self.group_num):
          sep_l = results_g[gi][b].extra_fields['labels']
          results_g[gi][b].extra_fields['labels'] = self.sep_labels_to_org_labels[gi][ sep_l ]
        result_b = [ results_g[gi][b] for gi in range(self.group_num) ]
        result_b = cat_boxlist_3d(result_b, per_example=False)
        result.append(result_b)

      #print(result[0].fields())
      return result


def show_box_fields(boxes, flag=''):
  print(f'\n\n{flag}')
  if isinstance(boxes, list):
    print(f'bs = {len(boxes)}')
    boxes = boxes[0]
  fields = boxes.fields()
  print(f'size:{len(boxes)} \nfields: {fields}')
  for fie in fields:
    fv = boxes.get_field(fie)
    fv_min = fv.min()
    fv_max = fv.max()
    print(f'{fie}: from {fv_min} to {fv_max}')

