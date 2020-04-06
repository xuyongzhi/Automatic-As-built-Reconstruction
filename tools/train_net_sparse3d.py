# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import open3d
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os, shutil
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference_3d import inference_3d
from maskrcnn_benchmark.engine.trainer_sparse3d import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from data3d.data import make_data_loader, check_data
from data3d.dataset_metas import DSET_METAS

def freeze_rpn_layers(model):
  for p in model.backbone.parameters():
    p.requires_grad = False
  for p in model.rpn.parameters():
    p.requires_grad = False
  #for p in model.roi_heads.parameters():
  #  p.requires_grad = True
  #model.roi_heads.requires_grad = False
  pass

def train(cfg, local_rank, distributed, loop, only_test, min_loss):
    ay = cfg.TEST.EVAL_AUG_THICKNESS_Y_TAR_ANC
    az = cfg.TEST.EVAL_AUG_THICKNESS_Z_TAR_ANC
    EVAL_AUG_THICKNESS = {'target_Y':ay[0], 'anchor_Y':ay[1],'target_Z':az[0], 'anchor_Z':az[1], }


    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    roi_only = cfg.MODEL.ROI__ONLY
    if roi_only:
      freeze_rpn_layers(model)

    optimizer = make_optimizer(cfg, model)

    arguments = {}
    arguments["iteration"] = 0
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed,
                  start_iter=arguments["iteration"])

    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, roi_only=roi_only
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    if only_test:
      return model, min_loss

    checkpoint_period = int(cfg.SOLVER.CHECKPOINT_PERIOD_EPOCHS * cfg.INPUT.Example_num / cfg.SOLVER.IMS_PER_BATCH)

    epochs_between_test = cfg.SOLVER.EPOCHS_BETWEEN_TEST
    loss_weights = cfg.MODEL.LOSS.WEIGHTS
    for e in range(epochs_between_test):
      min_loss = do_train(
          model,
          data_loader,
          optimizer,
          scheduler,
          checkpointer,
          device,
          checkpoint_period,
          arguments,
          e + loop * epochs_between_test,
          cfg.DEBUG.eval_in_train,
          output_dir,
          cfg.DEBUG.eval_in_train_per_iter,
          cfg.TEST.IOU_THRESHOLD,
          min_loss,
          eval_aug_thickness = EVAL_AUG_THICKNESS,
          loss_weights = loss_weights
      )
      pass

    return model, min_loss


def test(cfg, model, distributed, epoch):
    ay = cfg.TEST.EVAL_AUG_THICKNESS_Y_TAR_ANC
    az = cfg.TEST.EVAL_AUG_THICKNESS_Z_TAR_ANC
    EVAL_AUG_THICKNESS = {'target_Y':ay[0], 'anchor_Y':ay[1],'target_Z':az[0], 'anchor_Z':az[1], }

    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = [ make_data_loader(cfg, is_train=False, is_distributed=distributed) ]
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            dn = len(data_loaders_val[idx])
            iou_thr = int (10*cfg.TEST.IOU_THRESHOLD)
            aug_thickness = int(10*cfg.TEST.EVAL_AUG_THICKNESS_Y_TAR_ANC[0])
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_3d", dataset_name+f'_{dn}_iou_{iou_thr}_augth_{aug_thickness}')
            mkdir(output_folder)
            output_folders[idx] = output_folder
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference_3d(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN__ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            iou_thresh_eval = cfg.TEST.IOU_THRESHOLD,
            output_folder=output_folder,
            epoch = epoch,
            eval_aug_thickness = EVAL_AUG_THICKNESS,
        )
        synchronize()
    pass


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--only-test",
        dest="only_test",
        help="test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    intact_cfg(cfg)
    cfg.freeze()

    if cfg.MODEL.RPN__ONLY:
      args.skip_test = True
      cfg['DEBUG']['eval_in_train'] = -1
    #check_data(cfg)

    train_example_num = get_train_example_num(cfg)
    croi = '_CROI' if cfg.MODEL.CORNER_ROI else ''
    cfg['OUTPUT_DIR'] = f'{cfg.OUTPUT_DIR}_Tr{train_example_num}{croi}'
    if not cfg.MODEL.CLASS_SPECIFIC:
      cfg['OUTPUT_DIR'] += '_CA'
    if cfg.MODEL.RPN__ONLY:
      cfg['OUTPUT_DIR'] += '_RpnOnly'

    loss_weights = cfg.MODEL.LOSS.WEIGHTS
    if  loss_weights[4] > 0:
      k = int(loss_weights[4]*100)
      cfg['OUTPUT_DIR'] += f'_CorGeo{k}'
    if  loss_weights[5] > 0:
      k = int(loss_weights[5])
      p = int(loss_weights[6])
      cfg['OUTPUT_DIR'] += f'_CorSem{k}-{p}'
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
        cfn = os.path.basename(args.config_file)
        shutil.copyfile(args.config_file, f"{output_dir}/{cfn}")
        default_cfn = 'maskrcnn_benchmark/config/defaults.py'
        shutil.copyfile(default_cfn, f"{output_dir}/default.py")
        train_fns = 'data3d/suncg_utils/SuncgTorch/train_test_splited/train.txt'
        shutil.copyfile(train_fns, f"{output_dir}/train.txt")
        val_fns = 'data3d/suncg_utils/SuncgTorch/train_test_splited/val.txt'
        shutil.copyfile(train_fns, f"{output_dir}/val.txt")


    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    min_loss = 10000
    epochs_between_test = cfg.SOLVER.EPOCHS_BETWEEN_TEST
    for loop in range(cfg.SOLVER.EPOCHS // cfg.SOLVER.EPOCHS_BETWEEN_TEST):
      model, min_loss = train(cfg, args.local_rank, args.distributed, loop, args.only_test, min_loss)

      if not args.skip_test:
          test(cfg, model, args.distributed,
           epoch = (1+loop) * epochs_between_test - 1,
               )
          if args.only_test:
            break

def get_train_example_num(cfg):
    from data3d.data import DATASET_
    train_dataset = DATASET_('train', cfg)
    return len(train_dataset)

def intact_cfg(cfg):
  cfg.SPARSE3D.SCENE_SIZE = (np.array(cfg.SPARSE3D.VOXEL_FULL_SCALE).astype(np.float) / cfg.SPARSE3D.VOXEL_SCALE).tolist()
  intact_anchor(cfg)
  check_roi_parameters(cfg)
  intact_for_separate_classifier(cfg)
  intact_aug_thickness(cfg)

def intact_for_separate_classifier(cfg):
  dset_metas = DSET_METAS(cfg.INPUT.CLASSES)
  spec_classes_id = [[ dset_metas.class_2_label[c] for c in cs] for cs in cfg.MODEL.SEPARATE_CLASSES]
  spec_classes_id_flat = [c for cs in spec_classes_id  for c in cs ]

  separated_classes_flat = [c for cs in cfg.MODEL.SEPARATE_CLASSES for c in cs]
  cfg.MODEL.SEPARATE_CLASSES_ID = spec_classes_id
  remaining_classes = [c for c in cfg.INPUT.CLASSES if c not in separated_classes_flat]
  #cfg.MODEL.REMAIN_CLASSES = remaining_classes
  group_num = len(spec_classes_id)+1
  if len(spec_classes_id_flat) > 0:
    sep_r = 1.5 / group_num
    cfg.MODEL.RPN.FPN_PRE_NMS_TOP_N_TRAIN =   int(sep_r * cfg.MODEL.RPN.FPN_PRE_NMS_TOP_N_TRAIN)
    cfg.MODEL.RPN.FPN_PRE_NMS_TOP_N_TEST =    int(sep_r * cfg.MODEL.RPN.FPN_PRE_NMS_TOP_N_TEST)
    cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN =  int(sep_r * cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN)
    cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST =   int(sep_r * cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE= int(sep_r * cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG =  int(sep_r * cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG)

def intact_anchor(cfg):
  fpn_scalse = cfg.MODEL.RPN.RPN_SCALES_FROM_TOP
  strides = cfg.SPARSE3D.STRIDE
  nPlanesFront = cfg.SPARSE3D.nPlanesFront
  scales_selector_3d_2d = cfg.MODEL.RPN.RPN_3D_2D_SELECTOR
  anchor_size = cfg.MODEL.RPN.ANCHOR_SIZES_3D
  assert len(anchor_size) == len(scales_selector_3d_2d) == len(cfg.MODEL.RPN.USE_YAWS)
  assert len(cfg.MODEL.RPN.YAWS) == len(cfg.MODEL.RPN.RATIOS)

  scale_num = len(nPlanesFront)
  assert scale_num == len(strides) + 1
  ANCHOR_STRIDE = [np.array([1,1,1])]
  for s in range(scale_num-1):
    anchor_stride = ANCHOR_STRIDE[-1] * np.array(strides[s])
    ANCHOR_STRIDE.append(anchor_stride)
  anchor_stride = [ANCHOR_STRIDE[-i-1] for i in fpn_scalse]
  anchor_stride = anchor_stride + anchor_stride
  anchor_stride_final = [anchor_stride[i] for i in scales_selector_3d_2d]
  cfg.MODEL.RPN.ANCHOR_STRIDE = anchor_stride_final

  #cfg.MODEL.RPN.ANCHOR_STRIDE = list(reversed([ANCHOR_STRIDE[i] for i in fpn_scalse]))
  #cfg.MODEL.RPN.ANCHOR_SIZES_3D = list(reversed( cfg.MODEL.RPN.ANCHOR_SIZES_3D ))

  ns = len(fpn_scalse)
  na = len(anchor_size)
  #assert ns*2==na, f"fpn_scalse num {ns}*2 != anchor_size num {na}. The anchor size for each scale should be seperately"
  tmp = [i for i in scales_selector_3d_2d if i <ns]
  for s in range(1, len(tmp)):
    assert fpn_scalse[s-1] > fpn_scalse[s], "fpn should from level_id large (map large) to level_id small (map small)"
    assert anchor_size[s-1][0] < anchor_size[s][0], "ANCHOR_SIZES_3D should set from small to large, to match scale order of feature map"
    assert anchor_stride[s-1][0] < anchor_stride[s][0]

  #if len(anchor_size)>1:
  #  assert anchor_size[0][0] > anchor_size[1][0], "ANCHOR_SIZES_3D should set from small to large after reversed, to match scale order of feature map"

def intact_aug_thickness(cfg):
    pass
def check_roi_parameters(cfg):
  #spatial_scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES_SPATIAL
  #canonical_size = cfg.MODEL.ROI_BOX_HEAD.CANONICAL_SIZE
  #canonical_level = cfg.MODEL.ROI_BOX_HEAD.CANONICAL_LEVEL
  strides = cfg.SPARSE3D.STRIDE
  roi_scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES_FROM_TOP

  strides = np.array(strides)
  strides = np.cumprod(strides, 0)

  # RPN
  strides_ = np.flip(strides, 0)
  rpn_strides = strides_[cfg.MODEL.RPN.RPN_SCALES_FROM_TOP]
  full_scale = cfg.SPARSE3D.VOXEL_FULL_SCALE
  rpn_map_sizes = (np.array(full_scale).reshape(1,-1) / rpn_strides).astype(np.int32)

  cfg.MODEL.RPN.RPN_MAP_SIZES = rpn_map_sizes.tolist()

  # ROI
  spatial_scales_all = np.flip(1.0 / strides, 0)
  roi_spatial_scales = spatial_scales_all[roi_scales, :]
  assert np.all(roi_spatial_scales[:,0] == roi_spatial_scales[:,1])
  roi_spatial_scales_xy = roi_spatial_scales[:,0].tolist()

  cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES_SPATIAL = roi_spatial_scales_xy


  show = True
  if show:
    print(f"\n\nroi_spatial_scales_xy: {roi_spatial_scales_xy}\n")



if __name__ == "__main__":
    main()
