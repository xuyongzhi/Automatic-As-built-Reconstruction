# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    STEPS = [int(e * cfg.INPUT.Example_num / cfg.SOLVER.IMS_PER_BATCH)  for e in cfg.SOLVER.LR_STEP_EPOCHS ]
    STEPS = tuple(STEPS)
    WARMUP_ITERS = int(cfg.SOLVER.WARMUP_EPOCHS * cfg.INPUT.Example_num / cfg.SOLVER.IMS_PER_BATCH)
    WARMUP_ITERS = min(WARMUP_ITERS, 500)
    return WarmupMultiStepLR(
        optimizer,
        STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
