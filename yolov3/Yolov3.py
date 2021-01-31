import torch
import detectron2
from torch import nn
from .darknet import Darknet
from .loss import YOLOLossComputation
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.logger import log_first_n
from detectron2.structures import ImageList, Instances, Boxes
import logging
from detectron2.modeling.postprocessing import detector_postprocess
import sys
sys.path.append('./')
from yolov3.utils.boxes_op import non_max_suppression
@META_ARCH_REGISTRY.register()
class YOLOv3(nn.Module):
    def __init__(self, cfg):
        super(YOLOv3, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg
        self.darknet = Darknet(cfg)
        self.loss = YOLOLossComputation(cfg, self.darknet)
        # self.box_selector_test = YOLOPostProcessor(
        #
        # )
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preporcess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.cfg.MODEL.YOLOV3.SIZE_DIVISIBILITY)
        return images

    def forward(self, batched_inputs):
        if not self.training:
            results = self.inference(batched_inputs)
            return results
        images = self.preporcess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        yolo_out = self.darknet(images.tensor)
        losses = self.loss(yolo_out, gt_instances)
        return losses

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def inference(self, batched_inputs):
        images = self.preporcess_image(batched_inputs)
        predications, yolo_outs = self.darknet(images.tensor)
        dets = non_max_suppression(predications,
                                  conf_thres=self.cfg.MODEL.YOLOV3.TEST_CONF_TH,
                                  iou_thres=self.cfg.MODEL.YOLOV3.TEST_IOU_TH)
        results = []
        for im_i, det in enumerate(dets):
            inst = Instances(images.image_sizes[im_i])
            inst.pred_boxes = Boxes(det[:, :4])
            inst.pred_classes = det[:, 5]
            inst.scores = det[:, 4]
            results.append(inst)
        processed_results = YOLOv3._postprocess(results, batched_inputs, images.image_sizes)
        return processed_results