import json
import numpy as np
from matplotlib import pyplot as plt
from evaluation_scrip import group_by_key
def cat_recall_precision(pred, gt, iou_th):
    """
    :param pred: numpy array N * 4
    :param gt:   numpy array M * 4
    :param iou_th: IoU threshold
    :return: TPR FPR
    """
    pred_area = (pred[:, 2] - pred[:, 0] + 1) * ((pred[:, 3] - pred[:, 1] + 1))
    # print(gt.shape)
    gt_area = (gt[:, 2] - gt[:, 0] + 1) * ((gt[:, 3] - gt[:, 1] + 1))
    lt = np.maximum(pred[:, np.newaxis, :2], gt[:, :2])
    rb = np.minimum(pred[:, np.newaxis, 2:], gt[:, 2:])
    wh = (rb - lt + 1).clip(min=0)
    inter = (wh[:, :, 0] * wh[:, :, 1])
    union = pred_area[:, np.newaxis] + gt_area - inter
    iou = inter * 1. / union
    tp = 0
    fp = 0
    gt_check = np.zeros((gt.shape[0]))
    for i in range(pred.shape[0]):
        max_iou = np.max(iou[i])
        argmax_iou = np.argmax(iou[i])
        if max_iou > iou_th:
            if gt_check[argmax_iou] == 0:
                tp += 1
                gt_check[argmax_iou] = 1
            else:
                fp += 1
        else:
            fp += 1
    return tp, fp


def analysis_TPR_FPR(pred_json, gt_json, iou_th=0.1):
    pred_info = json.load(open(pred_json, 'r'))
    gt_info = json.load(open(gt_json, 'r'))
    cat_pred = group_by_key(pred_info, 'category_id')
    cat_gt = group_by_key(gt_info, 'category_id')
    num_points = int((1 - 0.01) / 0.01)
    for cat in cat_gt.keys():
        if cat not in cat_pred:
            print("[WARING] Category: {} Not in Prediction".format(cat))
            continue
        cat_gt_i = cat_gt[cat]
        cat_pred_i = cat_pred[cat]
        TPRs = []
        FPRs = []
        num_preds = len(cat_pred_i)
        for k in range(num_points):
            score_th = k * 0.01 + 0.01
            cat_gt_i_name = group_by_key(cat_gt_i, 'name')
            cat_pred_i_name = group_by_key(cat_pred_i, 'name')
            tp, fp = 0, 0
            for name in cat_pred_i_name.keys():
                if name not in cat_gt_i_name:
                    gt = np.asarray([[0, 0, 0, 0]])
                else:
                    gt = np.asarray([inst['bbox'] for inst in cat_gt_i_name[name]])
                pred = np.asarray([inst['bbox'] + [inst['score'], ] for inst in cat_pred_i_name[name]])
                candidate_ids = np.where(pred[:, -1] >= score_th)
                pred = pred[candidate_ids][:, :-1]
                tp_per_name, fp_per_name = cat_recall_precision(pred, gt, iou_th)
                tp += tp_per_name
                fp += fp_per_name
            # TPRs.append(tp * 1. / np.maximum(tp + fp, np.finfo(np.float64).eps))
            # FPRs.append(fp * 1. / np.maximum(tp + fp, np.finfo(np.float64).eps))
            TPRs.append(tp)
            FPRs.append(fp)
        plt.plot(TPRs, FPRs, 'm.-.')
        plt.show()


if __name__ == '__main__':
    gt_json = 'json_out/fcos_R_50_FPN_1x_steel_gt.json'
    pred_json = 'json_out/fcos_R_50_FPN_1x_steel.json'
    analysis_TPR_FPR(pred_json, gt_json)



