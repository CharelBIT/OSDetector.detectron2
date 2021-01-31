import cv2
import numpy as np
from detectron2.structures import Instances
def annos2submit(annos, file_name, DatasetName):
    assert isinstance(annos, list)
    results = []
    for anno in annos:
        ret = {}
        ret['name'] = file_name
        ret['category_id'] = anno['category_id']
        ret['segmentation'] = anno['segmentation']
        ret['bbox'] = anno['bbox']
        if "coco" in DatasetName:
            ret["bbox"] = [ret["bbox"][0], ret["bbox"][1],
                           ret["bbox"][0] + ret["bbox"][2],ret["bbox"][1] + ret["bbox"][3]]

        ret['score'] = anno['category_id']
        results.append(ret)
    return results

def inst2submit(insts, file_name):
    assert isinstance(insts, Instances)
    results = []
    for i in range(len(insts)):
        ret = {}
        ret['name'] = file_name
        ret['category_id'] = insts[i].get('pred_classes').cpu().numpy().tolist()[0]
        ret['score'] = insts[i].get('scores').cpu().numpy().tolist()[0]
        # print("[DEBUG] BBOX: {}".format(insts[i].get('pred_boxes')))
        ret['bbox'] = insts[i].get('pred_boxes').tensor.cpu().numpy().tolist()[0]
        if insts[i].has('pred_masks'):
            mask = insts[i].get('pred_masks').cpu().numpy()[0,...].astype(np.uint8)
        # print(mask.shape)
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                seg = contour.flatten().tolist()
                if len(seg) < 8:
                    continue
                segmentation.append(seg)
            ret['segmentation'] = segmentation
        results.append(ret)
    return results


