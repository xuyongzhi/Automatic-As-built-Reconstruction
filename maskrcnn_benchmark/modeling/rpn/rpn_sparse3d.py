# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from .loss_3d import make_rpn_loss_evaluator
from .anchor_generator_sparse3d import make_anchor_generator
from .inference_3d import make_rpn_postprocessor
from maskrcnn_benchmark.structures.bounding_box_3d import cat_scales_anchor, cat_boxlist_3d
from maskrcnn_benchmark.modeling.seperate_classifier import SeperateClassifier

DEBUG = 0
SHOW_TARGETS_ANCHORS = DEBUG and 0
SHOW_PRED_GT = DEBUG and 0
SHOW_ANCHORS_PER_LOC = DEBUG and 0

def cat_scales_obj_reg(objectness, rpn_box_regression, anchors):
  '''
     len(objectness) = len(rpn_box_regression) = scale num
     objectness[i].shape: [1,yaws_num,sparse_feature_num,1]
     rpn_box_regression[i].shape: [1,yaws_num*7,sparse_feature_num,1]
     anchors.batch_size() = batch size

     flatten order: [batch_size, scale_num, yaws_num, sparse_feature_num]

     rpn_box_regression_new: [yaws_num*sparse_feature_num of all scales,7]
     objectness_new: [yaws_num*sparse_feature_num of all scales]
  '''
  scale_num = len(objectness)
  assert scale_num == len(rpn_box_regression) == len(anchors)
  batch_size = anchors[0].batch_size()
  objectness_new = []
  rpn_box_regression_new = []
  for b in range(batch_size):
    objectness_new.append([])
    rpn_box_regression_new.append([])

  for s in range(scale_num):
    assert objectness[s].shape[0] == 1
    seperate_rpn = objectness[s].shape[-1]
    assert rpn_box_regression[s].shape[0] == 1
    assert rpn_box_regression[s].shape[-1] == 7 * seperate_rpn

    yaws_num = objectness[s].shape[2]
    examples_idxscope = anchors[s].examples_idxscope

    objectness_s = objectness[s].reshape(-1, seperate_rpn)
    rpn_box_regression_s = rpn_box_regression[s].reshape(-1,7*seperate_rpn)
    regression_flag = 'yaws_num_first'
    for b in range(batch_size):
      begin,end = examples_idxscope[b]
      obj = objectness_s[begin:end] # [yaws_num, sparse_feature_num]
      objectness_new[b].append( obj )
      reg = rpn_box_regression_s[begin:end] # [yaws_num*7, sparse_feature_num]

      #if regression_flag == 'yaws_num_first':
      #  reg = reg.view(yaws_num, 7, -1)
      #  reg = reg.permute(0,2,1)
      #elif regression_flag == 'boxreg_first':
      #  reg = reg.view(7, yaws_num, -1)
      #  reg = reg.permute(1,2,0)
      #else:
      #  raise NotImplementedError
      #reg = reg.reshape(-1,7)

      rpn_box_regression_new[b].append( reg )

  for b in range(batch_size):
    objectness_new[b] = torch.cat(objectness_new[b], 0)
    rpn_box_regression_new[b] = torch.cat(rpn_box_regression_new[b], 0)

  objectness_new = torch.cat(objectness_new, 0)
  rpn_box_regression_new = torch.cat(rpn_box_regression_new, 0)

  return objectness_new, rpn_box_regression_new


@registry.RPN_HEADS.register("SingleConvRPNHead_Sparse3D")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors_per_location):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors_per_location (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.num_anchors_per_location = num_anchors_per_location
        seperate_rpn = int(len(cfg.MODEL.SEPARATE_CLASSES) * cfg.MODEL.SEPARATE_RPN) + 1
        self.seperate_rpn = seperate_rpn
        self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=1, stride=1, padding=0  )
                #in_channels, in_channels, kernel_size=3, stride=1, padding=1  )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors_per_location * self.seperate_rpn, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors_per_location * 7 * self.seperate_rpn, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        reg_shape_method = 'box_toghter'
        #reg_shape_method = 'yaws_toghter'
        for feature in x:
            t = F.relu(self.conv(feature))    # [1,feature, sparse_locations, 1]
            # yaws_num: self.num_anchors_per_location
            logit = self.cls_logits(t) # [1,yaws_num, sparse_location_num, 1]
            logit = logit.permute(0,2,1,3) # [1,sparse_location_num, yaws_num,1]
            logit = logit.reshape(1, logit.shape[1], self.num_anchors_per_location, self.seperate_rpn)
            logits.append(logit)
            reg = self.bbox_pred(t) # [1,7*yaws_num, sparse_location_num,1]
            reg = reg.permute(0,2,1,3) # [1,sparse_location_num, yaws_num*7,1]
            if reg_shape_method == 'box_toghter':
              reg = reg.reshape(1,reg.shape[1],self.num_anchors_per_location,7*self.seperate_rpn) # [1,sparse_location_num, yaws_num,7]
            elif reg_shape_method == 'yaws_toghter':
              reg = reg.reshape(1,reg.shape[1],7,self.num_anchors_per_location) # [1,sparse_location_num, yaws_num,7]
              reg = reg.permute(0,1,3,2)
            else:
              raise NotImplementedError
            bbox_reg.append(reg)
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        in_channels = cfg.SPARSE3D.nPlaneMap
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()
        )

        rpn_box_coder = BoxCoder3D(is_corner_roi = False, weights=None)

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)
        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)
        class_specific = cfg.MODEL.CLASS_SPECIFIC
        self.seperate_classifier = SeperateClassifier(cfg.MODEL.SEPARATE_CLASSES_ID, len(cfg.INPUT.CLASSES), class_specific, 'RPN')

        self.box_coder = rpn_box_coder
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.add_gt_proposals = cfg.MODEL.RPN.ADD_GT_PROPOSALS

    def forward(self, inputs_sparse, features_sparse, targets=None):
        """
        Arguments:
            inputs_sparse (ImageList): inputs_sparse for which we want to compute the predictions
            features (list[Tensor]): features computed from the inputs_sparse that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        def reshape(f):
          return f.t().unsqueeze(0).unsqueeze(3)
        features = [fs.features for fs in features_sparse]
        features = [reshape(f) for f in features]
        #[print(f.shape) for f in features]
        # len(features) == levels_num
        # features[l]: [1,channels_num, n, 1]
        # n is a flatten of all the locations of all examples in a batch.
        # Because the feature map size in a batch may be diff for each example
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(inputs_sparse, features_sparse, targets)
        objectness, rpn_box_regression = cat_scales_obj_reg(objectness, rpn_box_regression, anchors)
        scale_num = len(anchors)
        anchors = cat_scales_anchor(anchors)
        anchors.constants['scale_num'] = scale_num
        anchors.constants['num_anchors_per_location'] = self.head.num_anchors_per_location

        device = features_sparse[0].features.device
        anchors = anchors.to(device)

        if SHOW_ANCHORS_PER_LOC:
          anchors.show_anchors_per_loc()

        if SHOW_TARGETS_ANCHORS:
            import numpy as np
            batch_size = len(targets)
            examples_scope = examples_bidx_2_sizes(inputs_sparse[0][:,-1])
            for bi in range(batch_size):
              se = examples_scope[bi]
              points = inputs_sparse[1][se[0]:se[1],0:3].cpu().data.numpy()
              print(f'\n targets')
              targets[bi].show(points=points)
              anchor_num = len(anchors)
              for i in np.random.choice(anchor_num, 5):
                anchor_i = anchors[int(i)].example(bi)
                print(f'\n anchor {i} / {anchor_num}')
                anchor_i.show__together(targets[bi], 200, points=points)
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass


        if self.training:
            boxes, loss = self._forward_train(anchors, objectness, rpn_box_regression, targets, debugs={'inputs_sparse': inputs_sparse})
        else:
            boxes, loss  = self._forward_test(anchors, objectness, rpn_box_regression, targets)

        if SHOW_PRED_GT:
          # len(boxes) = group num
          self.show_pred_gt(boxes, targets, inputs_sparse[1][:,:6])
        return boxes, loss

    def show_pred_gt(self, pred_boxes, targets, points):
        for pred in pred_boxes:
          pred.show_by_objectness(0.5,  points=points)
          #pred.show_by_objectness(0.97, targets[bi], points=points)
          pass

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets, debugs={}):
        if self.cfg.MODEL.RPN__ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
            if self.seperate_classifier.need_seperate:
              self.seperate_classifier.seperate_rpn_assin(targets)
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                if not self.seperate_classifier.need_seperate:
                  boxes = self.box_selector_train(anchors, objectness.squeeze(1),
                                rpn_box_regression, targets, self.add_gt_proposals)
                else:
                  boxes = self.seperate_classifier.seperate_rpn_selector(self.box_selector_train,
                                anchors, objectness, rpn_box_regression, targets, self.add_gt_proposals)

        if not self.seperate_classifier.need_seperate:
          loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
              anchors, objectness.squeeze(1), rpn_box_regression, targets, debugs
          )
          boxes.set_as_prediction()
          losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
          }
        else:
          loss_objectness, loss_rpn_box_reg = self.seperate_classifier.seperate_rpn_loss_evaluator(
                  self.loss_evaluator, anchors, objectness, rpn_box_regression, targets, debugs=debugs)
          gn = len(loss_objectness)
          losses = {}
          for gi in range(gn):
            if self.seperate_classifier.need_seperate:
              boxes[gi].set_as_prediction()
            losses[f"loss_objectness_{gi}"] = loss_objectness[gi]
            losses[f"loss_rpn_box_reg_{gi}"] = loss_rpn_box_reg[gi]

        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression, targets=None):
        if not self.seperate_classifier.need_seperate:
            boxes = self.box_selector_test(anchors, objectness.squeeze(1), rpn_box_regression, targets)
            boxes.set_as_prediction()
        else:
            boxes = self.seperate_classifier.seperate_rpn_selector(self.box_selector_test,
                            anchors, objectness, rpn_box_regression, targets, self.add_gt_proposals)
            boxes[0].set_as_prediction()
            boxes[1].set_as_prediction()
        if self.cfg.MODEL.RPN__ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            boxes = boxes.seperate_examples()
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
            #boxes = cat_boxlist_3d(boxes, per_example=True)
        return boxes, {}


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)


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

