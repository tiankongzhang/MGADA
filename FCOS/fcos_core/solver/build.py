# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, name):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if name == 'backbone':
            lr = cfg.SOLVER.BACKBONE.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BACKBONE.BASE_LR * cfg.SOLVER.BACKBONE.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'genbox':
            lr = cfg.SOLVER.GENBOX.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.GENBOX.BASE_LR * cfg.SOLVER.GENBOX.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'genfeature':
            lr = cfg.SOLVER.GENBOX.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.GENBOX.BASE_LR * cfg.SOLVER.GENBOX.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'fcos':
            lr = cfg.SOLVER.FCOS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.FCOS.BASE_LR * cfg.SOLVER.FCOS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'discriminator':
            lr = cfg.SOLVER.DIS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.DIS.BASE_LR * cfg.SOLVER.DIS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        else:
            raise AssertionError('here')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer

def make_optimizer_switch(cfg, model, name, istrain=False):
    if not istrain:
        for param in model.parameters():
            param.requires_grad = False
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    else:
        for param in model.parameters():
            param.requires_grad = True
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

    params = []
    for key, value in model.named_parameters():
        if name == 'backbone':
            lr = cfg.SOLVER.BACKBONE.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BACKBONE.BASE_LR * cfg.SOLVER.BACKBONE.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'genbox':
            lr = cfg.SOLVER.GENBOX.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.GENBOX.BASE_LR * cfg.SOLVER.GENBOX.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'genfeature':
            lr = cfg.SOLVER.GENBOX.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.GENBOX.BASE_LR * cfg.SOLVER.GENBOX.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'fcos':
            lr = cfg.SOLVER.FCOS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.FCOS.BASE_LR * cfg.SOLVER.FCOS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'discriminator':
            lr = cfg.SOLVER.DIS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.DIS.BASE_LR * cfg.SOLVER.DIS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        else:
            raise AssertionError('here')
        if not value.requires_grad:
            print(key, value.requires_grad)
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if len(params) == 0:
        optimizer = None
    else:
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer, name):
    if name == 'backbone':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.BACKBONE.STEPS,
            cfg.SOLVER.BACKBONE.GAMMA,
            warmup_factor=cfg.SOLVER.BACKBONE.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.BACKBONE.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.BACKBONE.WARMUP_METHOD,
        )
    elif name == 'genbox':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.GENBOX.STEPS,
            cfg.SOLVER.GENBOX.GAMMA,
            warmup_factor=cfg.SOLVER.GENBOX.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.GENBOX.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.GENBOX.WARMUP_METHOD,
        )
    elif name == 'genfeature':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.GENBOX.STEPS,
            cfg.SOLVER.GENBOX.GAMMA,
            warmup_factor=cfg.SOLVER.GENBOX.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.GENBOX.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.GENBOX.WARMUP_METHOD,
        )
    elif name == 'fcos':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.FCOS.STEPS,
            cfg.SOLVER.FCOS.GAMMA,
            warmup_factor=cfg.SOLVER.FCOS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.FCOS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.FCOS.WARMUP_METHOD,
        )
    elif name == 'discriminator':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.DIS.STEPS,
            cfg.SOLVER.DIS.GAMMA,
            warmup_factor=cfg.SOLVER.DIS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.DIS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.DIS.WARMUP_METHOD,
        )
    else:
        raise AssertionError('here')
