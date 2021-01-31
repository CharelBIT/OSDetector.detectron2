# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import pandas as pd
from collections import defaultdict
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from datasets import steel_buildin
from detectron2.data import DatasetCatalog, MetadataCatalog
from utils import annos2submit, inst2submit
from fcos.fcos import FCOS
from fcos.config import add_fcos_config

# constants
# WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_fcos_config(cfg)
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
        default="configs/wheat/fcos_R_50_FPN_1x_wheat.yaml",
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
    parser.add_argument('--test-path',
                        help='path to test dir',
                        default='/media/yons/gruntdata/data/WheatDetection/test',
                        type=str)
    parser.add_argument('--weight', help='Weight',
                        default='training_dir/fcos_R_50_FPN_1x_wheat_baseline/model_0008999.pth',
                        type=str)
    parser.add_argument(
        "--output",
        default='test_vis/',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
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
        default='submittion/fcos_baseline_conf0.5.csv',
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
    testlist = glob.glob(os.path.join(args.test_path, '*.jpg'))
    print(MetadataCatalog.get(cfg.DATASETS.TEST[0]))
    gt_info = []
    pred_info = []
    submit_info = pd.DataFrame(columns=['image_id', 'PredictionString'])
    template = '{} {} {} {} {}'
    for testfile in tqdm.tqdm(testlist, disable=not args.output):
        img_name = testfile.split('/')[-1]
        # use PIL, to be consistent with evaluation
        img = read_image(testfile, format="BGR")
        start_time = time.time()

        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                testfile,
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
                out_filename = os.path.join(args.output, img_name)
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)

        if args.save_result is not None:
            instances = predictions['instances']
            for i in range(len(instances)):
                submit = {}
                submit['image_id'] = img_name[:-4]
                bbox = instances[i].pred_boxes.tensor.cpu().tolist()[0]
                score = instances[i].scores.item()
                submit['PredictionString'] = template.format(score, bbox[0], bbox[1],
                                                             bbox[2] - bbox[0] + 1,
                                                             bbox[3] - bbox[1] + 1)
                submit_info = submit_info.append(submit, ignore_index=True)
            # gt_info.extend(annos2submit(test_dict['annotations'], test_dict['file_name']))
            # pred_info.extend(inst2submit(predictions['instances'], test_dict['file_name']))

    if args.save_result is not None:
        submit_info.to_csv(args.save_result, index=False)
        # import json
        # with open(args.save_result, 'w') as fp:
        #     json.dump(pred_info, fp, separators=(',', ': '), indent=4)
        # with open(args.save_result[:-5] + '_gt.json', 'w') as fp:
        #     json.dump(gt_info, fp, separators=(',', ': '), indent=4)


