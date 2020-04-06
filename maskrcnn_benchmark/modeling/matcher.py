# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

DEBUG = True
CHECK_SMAE_ANCHOR_MATCH_MULTI_TARGETS = DEBUG and False
CHECK_MISSED_TARGETS_NUM = DEBUG and False

ENALE_SECOND_THIRD_MAX__ONLY_HIGHEST_IOU_TARGET = False # reduce missed target
IGNORE_HIGHEST_MATCH_NEARBY = True
POS_HIGHEST_MATCH_NEARBY = False

class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False, yaw_threshold=3.1416*0.4):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        #assert yaw_threshold < 1.57
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches
        self.yaw_threshold = yaw_threshold

    def yaw_diff_constrain(self, match_quality_matrix, yaw_diff):
        if self.yaw_threshold > 1.58:
            return match_quality_matrix
        mask = torch.abs(yaw_diff) < self.yaw_threshold
        match_quality_matrix_new = match_quality_matrix * mask.float()
        return match_quality_matrix_new

    def __call__(self, match_quality_matrix, yaw_diff=None, flag='', cendis=None):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        if yaw_diff is not None:
          match_quality_matrix = self.yaw_diff_constrain(match_quality_matrix, yaw_diff)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix, cendis)

        if CHECK_MISSED_TARGETS_NUM:
            target_num = match_quality_matrix.shape[0]
            tmp = matches[matches>=0]
            detected_num = torch.unique(tmp).shape[0]
            missed_num = target_num - detected_num
            print(f'missed target num: {missed_num}  flag:{flag}')
        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix0, cendis=None):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        match_quality_matrix = match_quality_matrix0.clone()

        if ENALE_SECOND_THIRD_MAX__ONLY_HIGHEST_IOU_TARGET:
            # If one anchor has the maximum ious with multiple, some targets may
            # match no anchor.
            # An anchor can only be matched to the target, which has the largest iou
            # with it. As a result, a target may match the second or third ...
            # highest iou. This can guarantee every target not be missed.
            matched_vals_0, matches_0 = match_quality_matrix.max(dim=0)
            mask_only_max = match_quality_matrix*0
            tmp = torch.ones(matches_0.shape, device=matches_0.device)
            mask_only_max = mask_only_max.scatter(0, matches_0.view(1,-1), tmp.view(1,-1))
            match_quality_matrix *= mask_only_max

        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        #print(f'highest_quality_foreach_gt: \n{highest_quality_foreach_gt}')

        if cendis is None or True:
          gt_pred_pairs_of_highest_quality = torch.nonzero(
              match_quality_matrix == highest_quality_foreach_gt[:, None]
          )
        else:
          high_mask = match_quality_matrix >= highest_quality_foreach_gt[:, None]*0.95
          #gt_pred_pairs_of_highest_quality0 = torch.nonzero( )
          cendis1 = cendis + (1-high_mask.float()) * 1000
          cendis_min = cendis1.min(dim=1)[0]
          cendis_min_mask = cendis == cendis_min.view(-1,1)
          gt_pred_pairs_of_highest_quality = torch.nonzero(cendis_min_mask * high_mask)
          if not gt_pred_pairs_of_highest_quality.shape[0] == cendis.shape[0]:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass


        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

        if IGNORE_HIGHEST_MATCH_NEARBY:
            assert not POS_HIGHEST_MATCH_NEARBY
            ignore_threshold = highest_quality_foreach_gt - 0.05
            ignore_threshold = torch.max((ignore_threshold*0+1)*0.02, ignore_threshold)
            ignore_mask0 =  match_quality_matrix0 > ignore_threshold.view(-1,1)
            ignore_mask1 = ignore_mask0.any(dim=0)
            neg_mask = matches==Matcher.BELOW_LOW_THRESHOLD
            ignore_mask2 = ignore_mask1 * neg_mask
            ignore_ids = torch.nonzero(ignore_mask2).view(-1)
            matches[ignore_ids] = Matcher.BETWEEN_THRESHOLDS
        if POS_HIGHEST_MATCH_NEARBY:
            raise NotImplementedError
            assert not IGNORE_HIGHEST_MATCH_NEARBY
            ignore_threshold = highest_quality_foreach_gt - 0.05
            ignore_mask0 =  match_quality_matrix0 > ignore_threshold.view(-1,1)
            ignore_mask1 = ignore_mask0.any(dim=0)
            ignore_ids = torch.nonzero(ignore_mask1).view(-1)
            ignore_mask3 = torch.transpose( ignore_mask0[:,ignore_ids], 0, 1)
            gt_ids3 = torch.nonzero(ignore_mask3)
            matches[ignore_ids] = Matcher.BETWEEN_THRESHOLDS
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

        if CHECK_SMAE_ANCHOR_MATCH_MULTI_TARGETS:
            one_anchor_multi_targets = pred_inds_to_update.shape[0] - torch.unique(pred_inds_to_update).shape[0]
            if one_anchor_multi_targets >0:
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                pass


