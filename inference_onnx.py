import onnxruntime
import numpy as np
import cv2
import torch
from detectron2.data.transforms import ResizeShortestEdge
from yolov3.utils.boxes_op import xywh2xyxy, box_iou
import time
import torchvision
from detectron2.structures import Instances, Boxes
from detectron2.modeling.postprocessing import detector_postprocess
import copy

def show_insts(img, insts):
    boxes = insts.pred_boxes.tensor.cpu().numpy().astype(np.int)
    scores = insts.scores
    classes = insts.pred_classes
    template = '{}: {}'
    for i in range(boxes.shape[0]):
        img = cv2.rectangle(img, (boxes[i][0], boxes[i][1]),
                      (boxes[i][2], boxes[i][3]), color=(0, 0, 255))
        img = cv2.putText(img, template.format(classes[i], scores[i]),
                    color=(0, 0, 255),
                    org=(boxes[i][0], boxes[i][1]),
                    fontScale=1.,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    return img



def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = True  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction.shape[1] - 4  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    # Apply constraints
    box = xywh2xyxy(prediction[:, :4])
    if multi_label:
        i, j = (prediction[:, 4:] > conf_thres).nonzero().t()
        x = np.concatenate((box[i], prediction[i, j + 4][:, np.newaxis],
                            np.asarray(j, dtype=np.float)[:, np.newaxis]), axis=1)
        # x = torch.cat((box[i], prediction[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
    else:  # best class only
        # print()
        conf, j = prediction[:, 4:].max(1), prediction[:, 4:].argmax(1)
        x = np.concatenate((box, conf[:, np.newaxis], np.asarray(j, dtype=np.float)[:, np.newaxis]), 1)[conf > conf_thres]
    if classes:
        x = x[(j.reshape(-1, 1) == np.asarray(classes)).any(1)]

    n = x.shape[0]
    if not n:
        return np.zeros(shape=(0, 6))
    c = x[:, 5] * 0 if agnostic else x[:, 5]
    boxes, scores = copy.deepcopy(x[:, :4]) + c.reshape(-1, 1) * max_wh, x[:, 4]
    i = torchvision.ops.boxes.nms(torch.from_numpy(boxes),
                                  torch.from_numpy(scores),
                                  iou_thres)
    if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)

            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            # iou = iou * scores[None]
            weights = iou * scores[None]  # box weights
            x[i, :4] = (np.matmul(weights, x[:, :4])) / np.sum(weights, axis=1, keepdims=True)  # merged boxes
            # i = i[iou.sum(1) > 1]  # require redundancy
        # except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
        #     print(x, i, x.shape, i.shape)
        #     pass
    return x[i, ...]




def single_result(img_path, model):
    img = cv2.imread(img_path)
    origin_img = img[:, :, ::-1]
    trans_gen = ResizeShortestEdge([640, 640], 1024)
    height, width = origin_img.shape[:2]
    image = trans_gen.get_transform(img).apply_image(origin_img).transpose(2, 0, 1)[np.newaxis, ...]
    image = (image - np.asarray([103.530, 116.280, 123.675]).reshape(-1,3, 1, 1)) / \
            np.asarray([1.0, 1.0, 1.0]).reshape(-1,3, 1, 1)
    input = {model.get_inputs()[0].name: np.asarray(image, dtype=np.float32)}
    outputs = model.run(None, input)
    outputs[1] = outputs[1] * 640
    outputs = np.concatenate(outputs[::-1], axis=1)
    det = non_max_suppression(outputs)

    inst = Instances(image.shape[2:])
    inst.pred_boxes = Boxes(det[:, :4])
    inst.pred_classes = det[:, 5]
    inst.scores = det[:, 4]

    result = detector_postprocess(inst, height, width)
    print(result)
    img = show_insts(img, result)
    cv2.imwrite('result.png', img)
    return result

if __name__ == '__main__':
    onnx_path = 'fcos_base.onnx'
    model_session = onnxruntime.InferenceSession(onnx_path)
    input = np.zeros(shape=(3, 640, 640), dtype=np.float32)
    ort_inputs = {model_session.get_inputs()[0].name: input}
    outs = model_session.run(None, ort_inputs)
    print(outs)
    # img_path = '/media/yons/gruntdata/data/WheatDetection/train/81b76f372.jpg'
    # result = single_result(img_path, model_session)
    # print(outs[0])
    # print(outs[1]*640)
