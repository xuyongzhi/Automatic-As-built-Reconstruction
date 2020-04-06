# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DEBUG = CN()
_C.DEBUG.eval_in_train = 10
_C.DEBUG.eval_in_train_per_iter = -1

_C.MODEL = CN()
_C.MODEL.RPN__ONLY = False
_C.MODEL.ROI__ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "SparseRCNN"
_C.MODEL.CORNER_ROI = True
_C.MODEL.CLASS_SPECIFIC = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""

_C.MODEL.SEPARATE_CLASSES = []
# If false, only seperate ROI
_C.MODEL.SEPARATE_RPN = True
# -----------------------------------------------------------------------------
# Sparse 3D
# -----------------------------------------------------------------------------
_C.SPARSE3D = CN()
_C.SPARSE3D.VOXEL_SCALE = 50
_C.SPARSE3D.VOXEL_FULL_SCALE = [4096, 4096, 512]
_C.SPARSE3D.VAL_REPS = 3
_C.SPARSE3D.RESIDUAL_BLOCK = True
_C.SPARSE3D.BLOCK_REPS = 1
_C.SPARSE3D.nPlaneMap = 128
_C.SPARSE3D.nPlanesFront = [32, 64, 64, 128, 128, 128, 256, 256, 256]
_C.SPARSE3D.KERNEL = [[2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2],[2,2,2],[2,2,2],[2,2,2]]
_C.SPARSE3D.STRIDE = [[2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2],[2,2,2],[2,2,2],[2,2,2]]
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True

_C.INPUT.ELEMENTS = ['xyz', 'color', 'normal']
_C.INPUT.CLASSES = ['background', 'wall', 'door', 'window']
_C.INPUT.SCENES = []
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ("suncg_train", "suncg_val")
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ("suncg_test",)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 4
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "Sparse-R-50-FPN"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
#_C.MODEL.BACKBONE.OUT_CHANNELS = 128


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()


#_C.MODEL.RPN.ANCHOR3D_SIZES = (1,0.5,4, 3,0.5,4, 7,0.5,4)
# along yxz
_C.MODEL.RPN.ANCHOR_SIZES_3D = [[0.4,1.5,1.5],[1.5,1.5,1.0],[4,4,1.5],    [0.2,0.5,3], [0.4,1.5,3], [0.6,2.5,3]]
_C.MODEL.RPN.YAWS = (0, -1.57, -0.785, 0.785)
_C.MODEL.RPN.RATIOS = [[1,1,1],[1,2,1],[2,1,1],[1.7,1.7,1]]
# Enable use yaws or ratios for each scale separately
_C.MODEL.RPN.USE_YAWS = [1,1,1,  1,1,1]

_C.MODEL.RPN.USE_FPN = True
## Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
##_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
## Stride of the feature map that RPN is attached.
## For FPN, number of strides should match number of scales
##_C.MODEL.RPN.ANCHOR_STRIDE = [[8,8,729], [16,16,729], [32,32,729]] # NOT USED
## RPN anchor aspect ratios
##_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example) (->Matcher)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.55 #  0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example) (->Matcher)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.2 # 0.3
# Maximum yaw dif for positive anchor (->Matcher)
_C.MODEL.RPN.YAW_THRESHOLD = 0.7
# Total number of RPN examples per image (-> BalancedPositiveNegativeSampler)
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch (->BalancedPositiveNegativeSampler)
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.5 # 0.7
_C.MODEL.RPN.NMS_AUG_THICKNESS_Y_Z = [0.3,0.3]
_C.MODEL.RPN.LABEL_AUG_THICKNESS_Y_TAR_ANC = [0.4,0] # 0.4 is better than 06.
_C.MODEL.RPN.LABEL_AUG_THICKNESS_Z_TAR_ANC = [0.8,0] # 0.8 is better than 1
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep before applying NMS from all FPN
# levels
_C.MODEL.RPN.FPN_PRE_NMS_TOP_N_TRAIN = 2000 # 12000
_C.MODEL.RPN.FPN_PRE_NMS_TOP_N_TEST = 2000 # 6000
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 1000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 1000 #  2000
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead_Sparse3D"

# FROM_TOP: TOP 0 is the most sparse layer
_C.MODEL.RPN.RPN_SCALES_FROM_TOP =  [4,3,2,1]
# [indices of 3d, indeice of 2d]
_C.MODEL.RPN.RPN_3D_2D_SELECTOR =  [1,2,3, 4,5,6]
_C.MODEL.RPN.ADD_GT_PROPOSALS = True
# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = True
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (1.,1,1,1, 1,1,1) # (10., 10., 10, 5., 5., 5., 10)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 500
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.45 # 0.5
_C.MODEL.ROI_HEADS.NMS_AUG_THICKNESS_Y_Z = [0.2,0.2]
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 200
_C.MODEL.ROI_HEADS.LABEL_AUG_THICKNESS_Y_TAR_ANC = [0.4,0.4]
_C.MODEL.ROI_HEADS.LABEL_AUG_THICKNESS_Z_TAR_ANC = [0.6,0.6]


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FPNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = (5,11,4) #14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
#_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.5,0.25, 0.125)  # (1.0 / 16,)
#_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 512
_C.MODEL.ROI_BOX_HEAD.CANONICAL_SIZE = 8.0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES_FROM_TOP = (4,3)


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5


_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.YAW_MODE = 'Diff' #'SinDiff'
_C.MODEL.LOSS.WEIGHTS = [1.0,1,1,1,  0, 0, 0] # rpn_cls, rpn_reg, roi_cls,  roi_reg, cor_geo, cor_sem_pull, cor_sem_push
# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.BN_MOMENTUM = 0.95
_C.SOLVER.TRACK_RUNNING_STATS = True

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.LR_STEP_EPOCHS = (30,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 0.5
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD_EPOCHS = 20

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16


_C.SOLVER.EPOCHS = 100  # xyz add
_C.SOLVER.EPOCHS_BETWEEN_TEST = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# min iou for positive prediction in evaluation
_C.TEST.IOU_THRESHOLD = 0.2
_C.TEST.EVAL_AUG_THICKNESS_Y_TAR_ANC = [0.2,0.2]
_C.TEST.EVAL_AUG_THICKNESS_Z_TAR_ANC = [0.2,0.2]
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./RES"

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
