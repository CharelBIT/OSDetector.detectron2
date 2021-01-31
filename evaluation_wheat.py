import json
import numpy as np
from collections import defaultdict
def group_by_key(dicts, key):
    ret = defaultdict(list)
    for d in dicts:
        ret[d[key]].append(d)
    return ret

def cal_iou_matrix(bbox1, bbox2):
    bbox1_area = (bbox1[:, 3] - bbox1[:, 1] + 1) * (bbox1[:, 2] - bbox1[:, 0] + 1)
    bbox2_area = (bbox2[:, 3] - bbox2[:, 1] + 1) * (bbox2[:, 2] - bbox2[:, 0] + 1)
    ixmin = np.maximum(bbox1[:, np.newaxis, 0], bbox2[:, 0])
    iymin = np.maximum(bbox1[:, np.newaxis, 1], bbox2[:, 1])
    ixmax = np.minimum(bbox1[:, np.newaxis, 2], bbox2[:, 2])
    iymax = np.minimum(bbox1[:, np.newaxis, 3], bbox2[:, 3])
    iw = np.maximum(ixmax - ixmin + 1, 0)
    ih = np.maximum(iymax - iymin + 1, 0)
    inter = iw * ih
    union = bbox1_area[:, np.newaxis] + bbox2_area[np.newaxis, :] - inter
    iou_matrix = inter * 1. / union
    return iou_matrix

def cal_ap_for_single_image(gt, pred):
    pred.sort(key=lambda x: x['score'], reverse=True)
    pred_bboxes = np.asarray([inst['bbox'] for inst in pred])
    gt_bboxes = np.asarray([inst['bbox'] for inst in gt])
    iou_matrix = cal_iou_matrix(gt_bboxes, pred_bboxes)
    thresholds = list(np.arange(0.5, 0.75, step=0.05))
    ret = np.zeros(shape=(len(thresholds), 3), dtype=np.float) # tp, fp, fn
    for i, t in enumerate(thresholds):
        match_matrix = iou_matrix >= t
        pred_match = np.max(match_matrix, axis=0)
        ret[i, 0] = pred_match.sum()
        ret[i, 1] = len(pred) - pred_match.sum()

        gt_match = np.max(match_matrix, axis=1)
        ret[i, 2] = np.where(gt_match == 0)[0].shape[0]

    retVal = (ret[:, 0] / (ret.sum(axis=1))).mean()

    return retVal

def evaluation_wheat(gt_json, pred_json):
    gt_info = json.load(open(gt_json))
    pred_info = json.load(open(pred_json))
    gt_by_img = group_by_key(gt_info, 'name')
    pred_by_img = group_by_key(pred_info, 'name')
    image_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())
    results = []
    for key in image_ids:
        gt = gt_by_img[key]
        pred = pred_by_img[key]
        if (len(gt) == 0) ^ (len(pred) == 0):
            ret = 0.
        elif len(gt) == 0:
            ret = 1.
        else:
            ret = cal_ap_for_single_image(gt, pred)
        results.append(ret)
    print(np.asarray(results).mean())

if __name__ == '__main__':
    gt_json = 'json_out/fcos_R_50_FPN_1x_wheat_gt.json'
    pred_json = 'json_out/fcos_R_50_FPN_1x_wheat.json'
    evaluation_wheat(gt_json, pred_json)
