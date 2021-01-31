# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_fcos_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.FCOS = CN()
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.NORM_REG_TARGETS = False
    _C.MODEL.FCOS.CENTERNESS_ON_REG = False
    _C.MODEL.FCOS.USE_DCN_IN_TOWER = False
    _C.MODEL.FCOS.NUM_CONVS = 4

    _C.MODEL.FCOS.LOSS_GAMMA = 2.0
    _C.MODEL.FCOS.LOSS_ALPHA = 0.25
    _C.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
    _C.MODEL.FCOS.IOU_LOSS_TYPE = "iou"

    _C.MODEL.FCOS.PRIOR_PROB = 0.01
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]

    _C.MODEL.FCOS.INFERENCE_TH = 0.05
    _C.MODEL.FCOS.PRE_NMS_TOP_N = 1000
    _C.MODEL.FCOS.NMS_TH = 0.6
    _C.MODEL.FCOS.MASK_ON = False
    _C.MODEL.FCOS.TO_ONNX = False
    _C.MODEL.FCOS.BN_TYPE = "BN"

# add_fcos_config()
