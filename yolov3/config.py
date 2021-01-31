# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_yolo_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.YOLOV3 = CN()
    _C.MODEL.YOLOV3.CFG_FILE= 'cfg/yolov3-spp.cfg'
    _C.MODEL.YOLOV3.ONNX_EXPORT = False
    _C.MODEL.YOLOV3.NUM_CLASSES = 80

    _C.MODEL.YOLOV3.IOU_TH = 0.2
    _C.MODEL.YOLOV3.LOSS_REDUCE = 'mean'
    _C.MODEL.YOLOV3.CLS_WGT = 1.
    _C.MODEL.YOLOV3.OBJ_WGT = 1.
    _C.MODEL.YOLOV3.SMOOTH_EPS = 0.
    _C.MODEL.YOLOV3.FL_WGT = 0.
    _C.MODEL.YOLOV3.FL_GAMMA = 1.5
    _C.MODEL.YOLOV3.FL_ALPHA = 1.5
    _C.MODEL.YOLOV3.GIOU_RATIO = 1.

    _C.MODEL.YOLOV3.LOSS_GIOU_WGT = 3.54
    _C.MODEL.YOLOV3.LOSS_CLS_WGT = 37.4
    _C.MODEL.YOLOV3.LOSS_OBJ_WGT = 64.3

    _C.MODEL.YOLOV3.TEST_CONF_TH = 0.2
    _C.MODEL.YOLOV3.TEST_IOU_TH = 0.6
    _C.MODEL.YOLOV3.SIZE_DIVISIBILITY = 32

# add_fcos_config()
