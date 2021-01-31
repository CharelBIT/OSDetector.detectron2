# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from datasets import steel_buildin
from datasets import coco_custom_buildin
from detectron2.data import DatasetCatalog, MetadataCatalog
from utils import annos2submit, inst2submit
from fcos.fcos import FCOS
from yolov3.Yolov3 import YOLOv3
from fcos.config import add_fcos_config
from yolov3.config import add_yolo_config
# from detectron2.utils.fp16 import wrap_fp16_model
# constants
# WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_fcos_config(cfg)
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weight
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        # default="configs/yolov3.yaml",
        default='configs/mask_fcos_R_50_FPN_1x_steel.yaml',
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # parser.add_argument("--video-input", help="Path to video file.")
    # parser.add_argument(
    #     "--input",
    #     nargs="+",
    #     help="A list of space separated input images; "
    #     "or a single glob pattern such as 'directory/*.jpg'",
    # )
    parser.add_argument('--testlist-file',
                        help='path to testfile',
                        # default='/media/yons/gruntdata/data/WheatDetection/testlist.txt',
                        default='/gruntdata/data/severstal-steel-defect-detection/testlist.txt',
                        type=str)
    parser.add_argument('--data-root',
                        help='path to img root',
                        default='/gruntdata/data/severstal-steel-defect-detection/test_images/',
                        type=str)
    parser.add_argument('--weight', help='Weight',
                        default='training_dir/mask_fcos_R_50_FPN_1x_coco_steel_fp16/model_0016999.pth',
                        type=str)
    parser.add_argument(
        "--output",
        default='steel_mask_fcos/mask_fcos_R_50_FPN_1x_coco_steel',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--save-result',
        help='Save Instance Result',
        default='json_out/mask_fcos_R_50_FPN_1x_coco_steel.json',
        type=str
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # cfg.MODEL.WEIGHTS = 'output/model_final.pth'

    demo = VisualizationDemo(cfg)
    test_list = [line.strip()
                 for line in open(args.testlist_file).readlines()]

    test_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    print(MetadataCatalog.get(cfg.DATASETS.TEST[0]))
    gt_info = []
    pred_info = []
    for test_dict in tqdm.tqdm(test_dicts, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(os.path.join(test_dict['file_name']), format="BGR")
        start_time = time.time()
        path = test_dict['file_name'].split('/')[-1]

        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)

        if args.save_result is not None:
            gt_info.extend(annos2submit(test_dict['annotations'], test_dict['file_name'], cfg.DATASETS.TEST[0]))
            pred_info.extend(inst2submit(predictions['instances'], test_dict['file_name']))

    if args.save_result is not None:
        import json
        with open(args.save_result, 'w') as fp:
            json.dump(pred_info, fp, separators=(',', ': '), indent=4)
        with open(args.save_result[:-5] + '_gt.json', 'w') as fp:
            json.dump(gt_info, fp, separators=(',', ': '), indent=4)


