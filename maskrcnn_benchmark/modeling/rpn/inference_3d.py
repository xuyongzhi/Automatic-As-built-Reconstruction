# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D, cat_scales_anchor, cat_boxlist_3d
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_nms_3d
from maskrcnn_benchmark.structures.boxlist_ops_3d import remove_small_boxes3d

from ..utils import cat


DEBUG = True
SHOW_RPN_OUT_BEFORE_NMS = DEBUG and False
SHOW_NMS_OUT = DEBUG and False
SHOW_PRO_NUMS = DEBUG and False

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        batch_size,
        fpn_pre_nms_top_n,
        fpn_post_nms_top_n,
        nms_thresh,
        nms_aug_thickness,
        min_size,
        box_coder=None,
    ):
        """
        Arguments:
            fpn_pre_nms_top_n (int)
            fpn_post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
        """
        super(RPNPostProcessor, self).__init__()
        self.batch_size = batch_size
        self.fpn_pre_nms_top_n = fpn_pre_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.nms_thresh = nms_thresh
        self.nms_aug_thickness = nms_aug_thickness
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder()
        self.box_coder = box_coder

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: BoxList
            targets: BoxList
        """
        # Get the device we're operating on
        device = proposals.bbox3d.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]
        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
            gt_box.add_field("is_gt", torch.ones(len(gt_box), device=device))
            gt_box.constants = proposals.constants

        proposals.add_field("is_gt", torch.zeros(len(proposals), device=device))

        batch_size = proposals.batch_size()
        proposals = proposals.seperate_examples()
        proposals = [
              cat_boxlist_3d((proposal, gt_box), per_example=False)
              for proposal, gt_box in zip(proposals, gt_boxes)
          ]
        proposals = cat_boxlist_3d(proposals, per_example=True)

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: BoxList -> all examples within same batch are concated together
            objectness: tensor of size N
            box_regression: tensor of size N, 7
        """
        device = objectness.device
        assert objectness.shape[0] == box_regression.shape[0] == len(anchors)

        examples_idxscope = anchors.examples_idxscope
        batch_size = anchors.batch_size()
        result = []
        for bi in range(batch_size):
          # split examples in the batch
          s,e = examples_idxscope[bi]
          if SHOW_PRO_NUMS:
            print(f'\n\nRPN input anchor num: {e-s}')
          objectness_i0 = objectness[s:e]
          box_regression_i = box_regression[s:e]

          # put in the same format as anchors
          objectness_i1 = objectness_i0.sigmoid()

          # only choose top 2000 proposals for nms
          num_anchors = e-s
          fpn_pre_nms_top_n = min(self.fpn_pre_nms_top_n, num_anchors)
          objectness_i, topk_idx = objectness_i1.topk(fpn_pre_nms_top_n, dim=0, sorted=True)

          #batch_idx = torch.arange(N, device=device)[:, None]
          box_regression_i = box_regression_i[topk_idx]

          if anchors.size3d is None:
            pcl_size3d = None
          else:
            pcl_size3d = anchors.size3d[bi:bi+1]
          concat_anchors_i = anchors.bbox3d[s:e,:]
          concat_anchors_i = concat_anchors_i[topk_idx]

          # decode box_regression to get proposals
          proposals_i = self.box_coder.decode(
              box_regression_i, concat_anchors_i )

          #*********************************************************************
          # apply nms
          examples_idxscope_new = torch.tensor([[0, proposals_i.shape[0]]])
          boxlist = BoxList3D(proposals_i, pcl_size3d, mode="yx_zb",
                              examples_idxscope= examples_idxscope_new,
                              constants={'prediction':True})
          boxlist.add_field("objectness", objectness_i)
          boxlist.set_as_prediction()
          if SHOW_RPN_OUT_BEFORE_NMS:
            print(f'\n\n------------------------------------\n RPN out before NMS ')
            boxlist.show_together(targets[bi])
            boxlist.show_by_objectness(0.8, targets[bi])

          #boxlist = boxlist.clip_to_pcl(remove_empty=False)
          #boxlist = remove_small_boxes3d(boxlist, self.min_size)
          if SHOW_PRO_NUMS:
            print(f'before nms box num: {len(boxlist)}')
          boxlist_new = boxlist_nms_3d(
              boxlist,
              self.nms_thresh,
              nms_aug_thickness=self.nms_aug_thickness,
              max_proposals=self.fpn_post_nms_top_n,
              score_field="objectness",
              flag = 'rpn_post',
          )
          result.append(boxlist_new)
          if SHOW_PRO_NUMS:
            print(f'RPN out, after nms box num: {len(boxlist_new)}\n\n')

          if SHOW_NMS_OUT:
            print(f'\n\n------------------------------------\n RPN out after NMS ')
            print('inference_3d.py SHOW_NMS_OUT')
            objectness_i_new = boxlist_new.get_field('objectness')
            print(f"objectness: {objectness_i_new[0:10]}")
            boxlist_new.show_by_objectness(0.8, targets[bi])
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
        result = cat_boxlist_3d(result, per_example=True)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None, add_gt_proposals=False):
        """
        Arguments:
            anchors: BoxList
            objectness: tensor
            box_regression: tensor
            batch_size = anchors.batch_size()

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        #print(anchors.batch_size())
        boxlists = self.forward_for_single_feature_map(anchors, objectness, box_regression, targets)

        # append ground-truth bboxes to proposals
        if self.training and add_gt_proposals:
            assert targets is not None
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    fpn_pre_nms_top_n = config.MODEL.RPN.FPN_PRE_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_pre_nms_top_n = config.MODEL.RPN.FPN_PRE_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    nms_aug_thickness = config.MODEL.RPN.NMS_AUG_THICKNESS_Y_Z
    min_size = config.MODEL.RPN.MIN_SIZE
    batch_size = config.SOLVER.IMS_PER_BATCH
    box_selector = RPNPostProcessor(
        batch_size = batch_size,
        fpn_pre_nms_top_n=fpn_pre_nms_top_n,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        nms_aug_thickness = nms_aug_thickness,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
    )
    return box_selector
