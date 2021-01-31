import json
import numpy as np
from collections import defaultdict

def group_by_key(info, name):
    ret = defaultdict(list)
    for f in info:
        ret[f[name]].append(f)
        # ret[f[name]].append(f)
    return ret

def get_ap(recalls, precisions):
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    i = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap
def cat_pc(gt_info, pred_info, thresholds):
    num_gts = len(gt_info)
    image_gt = group_by_key(gt_info, 'name')
    image_gt_boxes = {name: np.asarray([inst['bbox'] for inst in insts])
            for name, insts in image_gt.items()}
    image_gt_checked = {name: np.zeros(shape=(len(insts), len(thresholds)))
            for name, insts in image_gt.items()
    }
    prediction = sorted(pred_info, key=lambda x: x['score'], reverse=True)
    nd = len(prediction)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(prediction):
        box = p['bbox']
        iou_max = -np.inf
        iou_argmax = -1
        try:
            gt_bboxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_bboxes = []
            gt_checked = None
        if len(gt_bboxes) > 0:
            ixmin = np.maximum(box[0], gt_bboxes[:, 0])
            iymin = np.maximum(box[1], gt_bboxes[:, 1])
            ixmax = np.minimum(box[2], gt_bboxes[:, 2])
            iymax = np.minimum(box[3], gt_bboxes[:, 3])
            iw = np.maximum(ixmax - ixmin + 1., 0)
            ih = np.maximum(iymax - iymin + 1., 0)
            inters = iw * ih
            union = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) + (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * \
            (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1) - inters
            iou = inters * 1. / union
            iou_max = np.max(iou)
            iou_argmax = np.argmax(iou)
        for t, threshold in enumerate(thresholds):
            if iou_max > threshold:
                if gt_checked[iou_argmax, t] == 0:
                    tp[i, t] = 1.
                    gt_checked[iou_argmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recall = tp / float(num_gts)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recall[:, t], precision[:, t])
    return recall, precision, ap
def evaluation_detection(pred_json, gt_json):
    pred_info = json.load(open(pred_json, 'r'))
    gt_info = json.load(open(gt_json, 'r'))

    cat_gt = group_by_key(gt_info, 'category_id')
    cat_pred = group_by_key(pred_info, 'category_id')
    cat_gt_list = sorted(cat_gt.keys())
    cat_pred_list = sorted(cat_pred.keys())
    print(cat_gt_list, cat_pred_list)
    thresholds = [0.1, 0.5, 0.75]
    aps = np.zeros((len(thresholds), len(cat_gt_list)))
    all_r, all_p, ap = cat_pc(gt_info, pred_info, thresholds)
    # result_per_class = []
    for i, cat in enumerate(cat_gt_list):
        r, p, ap = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)
        aps[:, i] = ap
    aps *= 100
    mAP = np.mean(aps, axis=1)
    print(aps)
    print(mAP)

if __name__ == '__main__':
    pred_json = 'json_out/yolo_dark_53_SPP_wheat.json'
    gt_json = 'json_out/yolo_dark_53_SPP_wheat_gt.json'
    evaluation_detection(pred_json, gt_json)
