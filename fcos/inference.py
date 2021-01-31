import torch
from detectron2.structures import Instances, Boxes, BoxMode
from detectron2.layers import batched_nms
from detectron2.utils.decorators import force_fp32
def clip_to_image(instance, remove_empty=True):
    TO_REMOVE = 1
    if not instance.has('pred_boxes'):
        return instance
    bboxes = instance.get('pred_boxes')
    h, w = instance.image_size[0], instance.image_size[1]
    bboxes.tensor[:, 0].clamp_(min=0, max=w - TO_REMOVE)
    bboxes.tensor[:, 1].clamp_(min=0, max=h - TO_REMOVE)
    bboxes.tensor[:, 2].clamp_(min=0, max=w - TO_REMOVE)
    bboxes.tensor[:, 3].clamp_(min=0, max=h - TO_REMOVE)
    # instance.set('bboxes', bboxes)
    if remove_empty:
        keep = (bboxes.tensor[:, 3] > bboxes.tensor[:, 1]) \
               & (bboxes.tensor[:, 2] > bboxes.tensor[:, 0])
        return instance[keep]
    return instance

def remove_small_boxes(instance, min_size):
    if not instance.has('pred_boxes'):
        return instance
    bboxes = instance.get('pred_boxes')
    if bboxes.tensor.size()[0] == 0:
        return instance
    xywh_bboxes = BoxMode.convert(bboxes.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    _, _, ws, hs = xywh_bboxes.unbind(dim=1)
    keep = (
            (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return instance[keep]


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
    @force_fp32(apply_to=['box_cls', 'box_regression', 'centerness'])
    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            instance = Instances((int(h), int(w)))
            bboxes = Boxes(detections)
            instance.set('pred_boxes', bboxes)
            instance.set('pred_classes', per_class)
            instance.set('scores', torch.sqrt(per_box_cls))
            # boxlist.add_field("labels", per_class)
            # boxlist.add_field("scores", torch.sqrt(per_box_cls))
            instance = clip_to_image(instance, remove_empty=True)
            instance = remove_small_boxes(instance, self.min_size)
            results.append(instance)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        instances = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(instance) for instance in instances]
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, instances):
        num_images = len(instances)
        results = []
        for i in range(num_images):
            # multiclass nms
            # result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            keep = batched_nms(instances[i].pred_boxes.tensor, instances[i].scores,
                               instances[i].pred_classes, self.nms_thresh)
            result = instances[i][keep]
            # result = instances[i]
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def build_fcos_inference_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMAGE
    bbox_aug_enabled = config.TEST.AUG.ENABLED

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=1,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
