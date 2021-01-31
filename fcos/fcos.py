import logging
import numpy as np
import math
import torch
from typing import List, Tuple
from torch import nn
from detectron2.structures import Instances,  pairwise_iou, Boxes
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList
from fcos.fcos_module import build_fcos_module
from detectron2.utils.logger import log_first_n
from detectron2.modeling.roi_heads import build_mask_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.layers import ShapeSpec
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.utils.events import get_event_storage

def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@META_ARCH_REGISTRY.register()
class FCOS(nn.Module):
    def __init__(self, cfg):
        super(FCOS, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.fcos_module = build_fcos_module(cfg, self.backbone.output_shape())
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

        self.ft = torch.half if cfg.SOLVER.FP16.ENABLED else torch.float
        self.mask_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self._init_mask_head(cfg, self.backbone.output_shape())

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.to_onnx = cfg.MODEL.FCOS.TO_ONNX

        self.to(self.device)

        # self.mask_on = cfg.MODEL.FCOS.MASK_ON
        # if self.mask_on:

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        if not self.mask_on:
            return None
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.mask_in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.mask_in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def preporcess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images.tensor = images.tensor.to(self.ft)
        return images

    def forward(self, batched_inputs):
        if self.to_onnx and self.training != True and isinstance(batched_inputs, torch.Tensor):
            return self.forward_onnx(batched_inputs)
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
        features = self.backbone(images.tensor)
        boxes, losses = self.fcos_module(images, features, gt_instances)
        loss_mask = {}
        if self.mask_head is not None:
            proposals = boxes
            for proposal in proposals:
                proposal.proposal_boxes = proposal.pred_boxes
                proposal.remove('pred_boxes')
                proposal.gt_classes = proposal.pred_classes
                proposal.remove('pred_classes')
            proposals = self.add_ground_truth_to_proposal_and_matcher(proposals, gt_instances)
            features = [features[f] for f in self.mask_in_features]
            proposals, _ = select_foreground_proposals(proposals, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_features = mask_features.to(self.ft)
            loss_mask = self.mask_head(mask_features, proposals)
        losses.update(loss_mask)
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
        features = self.backbone(images.tensor)
        results, _ = self.fcos_module(images, features)
        if self.mask_head is not None:
            pred_boxes = [x.pred_boxes for x in results]
            features = [features[f] for f in self.mask_in_features]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_features = mask_features.to(self.ft)
            results = self.mask_head(mask_features, results)
        processed_results = FCOS._postprocess(results, batched_inputs, images.image_sizes)
        return processed_results
    @torch.no_grad()
    def add_ground_truth_to_proposal_and_matcher(self, proposals: List[Instances], gt_instances: List[Instances]):
        assert len(proposals) == len(gt_instances)
        device = proposals[0].scores.device
        new_proposals = []
        for proposal, gt_instance in zip(proposals, gt_instances):
            if len(gt_instance) == 0:
                continue
            instance = Instances(image_size=proposal.image_size)
            instance.proposal_boxes = gt_instance.gt_boxes.to(device)
            instance.gt_classes = gt_instance.gt_classes.to(device)
            score_val = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
            instance.scores = (score_val * torch.ones(len(gt_instance))).to(device)
            proposal = Instances.cat([proposal, instance])
            new_proposals.append(proposal)

        num_fg_samples = []
        num_bg_samples = []
        proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(new_proposals, gt_instances):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

            # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        return proposals_with_gt

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    def forward_onnx(self, batched_inputs):
        with torch.no_grad():
            if len(self.pixel_std.size()) == 3:
                self.pixel_std = self.pixel_std[None, ...]
                self.pixel_mean = self.pixel_mean[None, ...]
            # self.pixel_mean.detach()
            image = self.normalizer(batched_inputs)
            feature = self.backbone(image)
            box_cls, box_regression, centerness = self.fcos_module(image, feature)
            return box_cls, box_regression, centerness



