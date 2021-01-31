import torch
from collections import OrderedDict
if __name__ == '__main__':
    yolo_model = '/home/yons/PycharmProjects/yolov3/weights/last.pt'
    detectron_model = 'training_dir/dectron2_yolo.pt'

    ckpt = torch.load(yolo_model)['model']
    new_ckpt = OrderedDict()
    for key in ckpt.keys():
        print("[INFO] convert key: {}".format(key))
        new_ckpt['darknet.' + key] = ckpt[key]
    torch.save(new_ckpt, detectron_model)