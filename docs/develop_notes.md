# xyz


# on going process
- The corner net does not work for small objects
- improve loss for 2 corners regression: order issue
- write torch version: bbox3d_ops_torch.py: from_2corners_to_yxzb, from_yxzb_to_2corners

- A mistake in SUNCG_METAS: 'ceiling':5, 'floor': 4, change to 'ceiling':4, 'floor': 5 later.
        Modify suncg_eval.py / get_obj_nums as well 
- Test effect of RPN number: speed and precision trade-off
- Focal loss
- - RPN ROI loss weights
- upgrade anchor policy, IOU is not good, centroid distance should be considered
- update c++ version 3d nms 
- Considering put ceiling, floor and room to a separated classifier. Because the anchor policy is different, shared head may not satisfy.
- For ceiling, floor, when length == thickness/width, yaw is confusing
- try fpn321, to accelerate training speed
- iou criterion problem
- good rpn proposals are recognized as false negative, apply aug in iou of roi
- 3d box encoding is not right: /home/z/Research/Detection_3D/second/pytorch/core/ box_torch_ops.py
- (1) one anchor mathch two objects, (2) For long objects, a low iou is matched, while other close low iou is neg
- clip_to_pcl in bounding_box_3d.py not implemented
- considering aug both proposal and targets in match_targets_to_proposals in roi_heads/box_head_3d/loss.py
- \_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
- sparseconvnet/tools_3d_2d.py when not dense, 0 used, if this is ok?
- subsample not understand:   
        (2)  modeling/roi_heads/box_head_3d/box_head.py: self.loss_evaluator.subsample(proposals, targets)
- add yaw loss
- rethink how to improve acc for long wall: add yaw loss
- ** Big object is really hard ** crop gt box with anchor
- multi scale: (1) residual (2) feature concate in each scale of pyramid (3) feature concate to one single layer
- IOU: area_inter / (area2 + max(0,area1*0.5 - area_inter))   
   second/core/non_max_suppression/nms_gpu.py devRotateIoUEval
- ** Decormable net/cnn ** avoid rectangle conv kernel and fixed net. use adaptive formable net. use deformable cnn. 

# Geometric

## Box definition
```
geometric_util.py/OBJ_DEF
geometric_torch.py/OBJ_DEF/limit_yaw, check_bboxes
utils3d/bbox3d_ops.py/Bbox3D
```
- standard 3d box:   
```
'standard': [xc, yc, zc, x_size, y_size, z_size, yaw_s]
up_axis='Z', make always  x_size > y_size, y_size is thickness
yaw_s: [0, pi] 
bbox3d_ops.py/limit_yaw
```
- yx_zb 3d box:
```
'yx_zb':    [xc, yc, z_bot, y_size, x_size, z_size, yaw_s-0.5pi]
up_axis='Z', make always  x_size > y_size, y_size is thickness
yaw_s-0.5pi:   [-pi/2, pi/2]  
```
  data preparation: standard  
  feed into network: yx_zb (to use second lib)  

- yaw positive direction:
```
(1) clock wise during data preparation and rpn.
ref: utils3d/bbox3d_ops.py Bbox3D.bbox_corners()  check by: review_bbox_format()  
Although in geometric_util.py, positive for Rz is anti-clock wise, by do not transposing R in Bbox3D.bbox_corners(), it is clock wise finanly.  
(2) anti-clock wise in ROIAlignRotated_cuda.cu/ROIAlignRotated_forward_cuda
ref: layers/roi_align_rotated_3d.py  
This is the definition in caff2, so keep it. change sign of yaw in roi_align_rotated_3d.py/ROIAlignRotated3D.forward()
```

## box encoding
- modeling/box_coder_3d.py/encode & decode  
- second.pytorch.core.box_torch_ops/second_box_encode & second_box_decode  
- box_torch_ops.py/second_box_encode
smooth_dim = True  
```
lt = lg / la - 1
wt = wg / wa - 1
ht = hg / ha - 1
rt = rg - ra
```

## Taregts
- maskrcnn_benchmark/modeling/rpn/loss_3d.py: prepare_targets / self.box_coder.encode
- maskrcnn_benchmark/modeling/roi_heads/box_head_3d/loss.py: prepare_targets / self.box_coder.encode

## Yaw loss
- layers/smooth_l1_loss.py/get_yaw_loss
1. Dif: abs(yaw_t - yaw_p)
2. Dif_sin: abs(sin(yaw_t - yaw_p))
3. Min_dif_sin: min(1,2)
Dif cannot understand -pi/2==pi/2.   
Dif_sin does not limit in [-pi/2, pi/2].

## data preparation
- pcl input normaliztion   
```
data3d/data.py trainMerge
pcl xyz: [0:max]
```

# Data generation

# Data generation steps for as-built BIM

-  data3d/suncg.py/parse_house()
-  data3d/suncg.py/gen_train_eval_split()
-  data3d/suncg.py/gen_house_names_1level()

-  data3d/indoor_data_util.py/creat_splited_pcl_box()
-  data3d/indoor_data_util.py/creat_indoor_info_file()

* crop_bbox_by_points=False in bbox3d_ops.py/Bbox3D.crop_bbox_by_points
* keep_unseen_intersection=False in indoor_data_util.py/IndoorData.split_bbox

# run
- run.sh
- load test resut:   
  maskrcnn_benchmark/engine/inference_3d.py: inference_3d/load_pred

# Debug
 - data3d/evaluation/suncg/suncg_eval.py:  
        SHOW_PRED  
        DEBUG_DATA_SAMPLE 
 - sparseconvnet/fpn_net.py: SHOW_MODEL

 - modeling/rpn/rpn_sparse3d.py 
        SHOW_TARGETS_ANCHORS  
        SHOW_PRED_GT  
        SHOW_ANCHORS_PER_LOC    

 - modeling/rpn/loss_3d.py  
        CHECK_MATCHER
        SHOW_POS_ANCHOR_IOU_SAME_LOC: the positive anchor policy  
        SHOW_IGNORED_ANCHOR  
        SHOW_POS_NEG_ANCHORS  
        SHOW_PRED_POS_ANCHORS  

 - rpn/anchor_generator_sparse3d.py  AnchorGenerator/forward:  
        SHOW_ANCHOR_EACH_SCALE:

 - rpn/inference_3d.py
        SHOW_RPN_OUT_BEFORE_NMS  
        SHOW_NMS_OUT  
 - engine/inference_3d.py  
        LOAD_PRED
 - roi_heads/box_head_3d/inference.py  
        MERGE_BY_CORNER

# configurations:
- maskrcnn_benchmark/config/defaults.py 
- configs/sparse_faster_rcnn.yaml

## Learning rate
- maskrcnn_benchmark/solver/lr_scheduler.py

## IOU augmentation
* Aug thickness befor iou for targets preparation only. Do not aug thickness for NMS.
* * 
### RPN TARGET
- \_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.55
- \_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.25
    /home/z/Research/Detection_3D/maskrcnn_benchmark/modeling/rpn/loss_3d.py/make_rpn_loss_evaluator  
    ~/Research/Detection_3D/maskrcnn_benchmark/modeling/matcher.py
- \_C.MODEL.RPN.AUG_THICKNESS_TAR_ANC = [0.3,0] 
    /home/z/Research/Detection_3D/maskrcnn_benchmark/modeling/rpn/loss_3d.py/match_targets_to_anchors
### RPN NMS
- \_C.MODEL.RPN.NMS_THRESH  
    /home/z/Research/Detection_3D/maskrcnn_benchmark/modeling/rpn/inference_3d.py/forward_for_single_feature_map/boxlist_nms_3d  
    2d iou  
    no box aug  
### ROI TARGET
- \_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
- \_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
    modeling/roi_heads/box_head_3d/loss.py/make_roi_box_loss_evaluator  
    ~/Research/Detection_3D/maskrcnn_benchmark/modeling/matcher.py  
- \_C.MODEL.ROI_HEADS.AUG_THICKNESS_TAR_ANC = [0.2,0.2]
        maskrcnn_benchmark/modeling/roi_heads/box_head_3d/loss.py/match_targets_to_proposals  
### ROI NMS
- \_C.MODEL.ROI_HEADS.NMS   = 0.3
        ~/Research/Detection_3D/maskrcnn_benchmark/structures/boxlist_ops_3d.py  
        modeling/roi_heads/box_head_3d/inference.py
        This is 2d iou.   
        No box aug
### TEST
- \_C.TEST.IOU_THRESHOLD = 0.1
    suncg_eval.py/calc_detection_suncg_prec_rec     
    no box augmentation    
    criterion = -1

# Basic code structure
- maskrcnn_benchmark/structures/bounding_box_3d.py/BoxList3D
        Box class used for training
## MODEL
1. tools/train_net_sparse3d.py:main -> :train & test
2. modeling/detector/detectors.py: 
```
build_detection_model -> sparse_rcnn.py:SparseRCNN  
In SparseRCNN:  
features = self.backbone(points)  
proposals, proposal_losses = self.rpn(points, features, targets)  
x, result, detector_losses = self.roi_heads(features, proposals, targets)  
```
3. modeling/backbone/backbone.py:
```
build_backbone -> :build_sparse_resnet_fpn_backbone -> sparseconvnet.FPN_Net  
```
4. modeling/rpn_sparse3d.py: 
```
build_rpn ->  RPNModule -> inference_3d/make_rpn_postprocessor -> loss_3d/make_rpn_loss_evaluator  
```
4.1 RPNModule
```
objectness, rpn_box_regression = self.head(features)  
anchors = self.anchor_generator(points_sparse, features_sparse)  
-> rpn/anchor_generator_sparse3d.py/AnchorGenerator.forward()
```
4.2 modeling/rpn/loss_3d.py:
```
make_rpn_loss_evaluator -> RPNLossComputation  
objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(...)  
box_loss = smooth_l1_loss(...)  
```
4.3 modeling/rpn/inference_3d.py:
```
make_rpn_postprocessor -> RPNPostProcessor -> structures.boxlist3d_ops.boxlist_nms_3d  
-> second.pytorch.core.box_torch_ops.rotate_nms & multiclass_nms + second.core.non_max_suppression.nms_gpu/rotate_iou_gpu_eval
```
5. roi: 
** process **
```
(1) modeling/detector/sparse_rcnn.py/SparseRCNN: 
        x, result, detector_losses = self.roi_heads(features, proposals, targets)
(2) modeling/roi_heads/box_head_3d/box_head.py ROIBoxHead3D
(3) roi_heads/box_head_3d/roi_box_feature_extractors.py
(4) modeling/poolers.py 
```
** POOLER_SCALES_SPATIAL **
```
Automatically calculated in train_net_sparse3d.py/check_roi_parameters by strides
roi_box_feature_extractors.py -> poolers_3d.py/LevelMapper_3d -> layers/roi_align_rotated_3d.py/ROIAlignRotated3D

rate = size / canonical_size
just find the closest level of feature map with rate
* size: (1) The square root of predicted box area. Used in MaskRCNN 
        (2) The maximum of width and length. USed in this project.
* canonical_size: the canonical size of object in the full size feature map (original point cloud). For example, 4 meters when pcl size is 8 meters.
```
** POOLER Unit **
```
(1) The original proposal box unit is meter. Convert to pixel before feed into ROIALIGN:
roi_box_feature_extractors.py/convert_metric_to_pixel
(2) The features froom backbone is sparse3d. Convert to dense in roi_align_rotated_3d.py/ROIAlignRotated3D/sparse_3d_to_dense_2d
```


## matcher
1. rpn/loss_3d.py/make_rpn_loss_evaluator
```
cfg.MODEL.RPN.FG_IOU_THRESHOLD
cfg.MODEL.RPN.BG_IOU_THRESHOLD
allow_low_quality_matches=True
yaw_threshold = cfg.MODEL.RPN.YAW_THRESHOLD
```

## BalancedPositiveNegativeSampler
1. rpn/loss_3d.py/make_rpn_loss_evaluator
```
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
cfg.MODEL.RPN.POSITIVE_FRACTION
```

## loss
- rpn_sparse3d.py: loss_objectness, loss_rpn_box_reg
- modeling/roi_heads/box_head_3d/box_head.py:  loss_classifier_roi, loss_box_reg_roi

## Data feeding
1. data.py/make_data_loader/

## Evaluation
- maskrcnn_benchmark/data/datasets/evaluation/suncg/suncg_eval.py/do_suncg_evaluation

## Anchor
- rpn/anchor_generator_sparse3d.py/AnchorGenerator.forward()
* ANCHOR_SIZES_3D: [[0.5,1,3], [1,4,3]]
* YAWS: (0, -1.57, -0.785, 0.785)
- BG_IOU_THRESHOLD: 0.1
- FG_IOU_THRESHOLD: 0.3

- flatten order
```
1. anchor_generator_sparse3d.py/AnchorGenerator.grid_anchors:   
        flatten order: [sparse_location_num, yaws_num, 7]     
2. rpn_sparse3d.py/RPNModule/forward ->  bounding_box_3d.py/ cat_scales_anchor:   
        final flatten order: [batch_size, scale_num, sparse_location_num, yaws_num]
3. loss_3d.py/RPNLossComputation.prepare_targets:  
        labels same as anchors   
4. objectness and rpn_box_regression  
        rpn_sparse3d.py/RPNHead.forward: [sparse_location_num, yaws_num]
                reg_shape_method = 'box_toghter' or 'yaws_toghter'  
        rpn_sparse3d.py/cat_scales_obj_reg:         
                flatten order same as anchor
```

### Positive policy **Very Important**
-1:ignore, 0: negative, 1:positive  
Positive anchor: 1. this anchor location is the closest to the target centroid. 2. the feature receptive field contains the target at most.
```
cfg.MODEL.RPN.FG_IOU_THRESHOLD
cfg.MODEL.RPN.BG_IOU_THRESHOLD
cfg.MODEL.RPN.YAW_THRESHOLD
```
- modeling/rpn/loss_3d.py/RPNLossComputation/match_targets_to_anchors:  
        match_quality_matrix = boxlist_iou_3d(anchor, target)  
        matched_idxs = self.proposal_matcher(match_quality_matrix)  
- second.core.non_max_suppression.nms_gpu/rotate_iou_gpu_eval &  devRotateIoUEval:   
        criterion == 2:  area_inter / (area2 + max(0,area1*0.5 - area_inter)), area2 is target  
- modeling/matcher.py/Matcher/__call__ & yaw_diff_constrain
- modeling/balanced_positive_negative_sampler.py

###  model classes
```
- SparseRCNN:  maskrcnn_benchmark/modeling/detector/sparse_rcnn.py
- RPNModule: maskrcnn_benchmark/modeling/rpn/rpn_sparse3d.py
- RPNPostProcessor: maskrcnn_benchmark/modeling/rpn/rpn_sparse3d.py
```

### maskrcnn_benchmark call second
```
- maskrcnn_benchmark/structures/boxlist3d_ops.py:
        from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval
        from second.pytorch.core.box_torch_ops import rotate_nms

- maskrcnn_benchmark/modeling/box_coder_3d.py
        from second.pytorch.core.box_torch_ops import second_box_encode, second_box_decode
```

### maskrcnn_benchmark call sparse_faster_rcnn
```
- modeling/backbone/backbone.py/build_sparse_resnet_fpn_backbone:
        fpn = scn.FPN_Net(full_scale, dimension, raw_elements, block_reps, nPlanesF,...)
```

## add_gt_proposals
- maskrcnn_benchmark/modeling/rpn/inference_3d.py:RPNPostProcessor/forward
- modeling/roi_heads/box_head_3d/box_head.py:ROIBoxHead3D -> rm_gt_from_proposals
- modeling/roi_heads/box_head_3d/loss.py: show_roi_cls_regs    

# Ideas for the future
- 3D object detection by keypoint
- 3D object detection with deformable convolution
- BIM detection aided by constrain of connection relationship
- Indoor navigation with 3d mapping of BIM 
