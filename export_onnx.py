import torch
from torch.autograd import Variable
from yolov3.config import add_yolo_config
from detectron2.config import get_cfg
from fcos.fcos import FCOS
from yolov3.Yolov3 import YOLOv3
from detectron2.checkpoint import DetectionCheckpointer
from fcos.config import add_fcos_config
from detectron2.modeling import build_model

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_fcos_config(cfg)
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
if __name__ == '__main__':
    config_file = 'configs/mask_fcos_R_50_FPN_headBN_1x_steel.yaml'
    model_path = 'training_dir/mask_fcos_R_50_FPN_headBN_1x_coco_steel_fp16/model_0003999.pth'

    cfg = get_cfg()
    add_fcos_config(cfg)
    cfg.merge_from_file(config_file)
    model = build_model(cfg).to('cuda:0')
    model.eval()
    input = torch.zeros(size=(1,3, 640, 640), dtype=torch.float).to(model.device)
    DetectionCheckpointer(model=model).load(model_path)
    box_cls, box_regression, centerness = model(input)
    print(box_cls, box_regression, centerness)
    # confes = outputs[0, :, 4, None] * outputs[0, :, 5:]
    # bboxes = outputs[0,:,:4]
    # confes = torch.cat(confes, dim=0)
    assert len(box_cls) == len(box_regression) == len(centerness)
    for i in range(len(box_cls)):
        print("[INFO]box cls shape: {}, box regression shape: {}, centerness shape: {}".format(box_cls[i].size(),
                                                                                               box_regression[i].size(),
                                                                                               centerness[i].size()))

    input = Variable(torch.zeros(size=(3, 640, 640))).cuda()
    torch.onnx.export(model, input, 'fcos_base_0003999.onnx', verbose=True)
