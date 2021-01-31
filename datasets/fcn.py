import os
import imagesize
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ["get_fcn_dict",]


def get_fcn_dict(img_dir, label_dir, datalist_file, dataset_name):
    def _get_annos(label_file):
        info = [line.strip() for line in open(label_file).readlines()]
        num_inst = int(info[1])
        annos = []
        for i in range(num_inst):
            anno = {}
            label = thing_dataset_id_to_contiguous_id[int(info[i * 3 + 2])]
            # label = int(info[i * 3 + 2])

            if label < 0 or label > name_class:
                continue
            bbox = list(map(int, info[i * 3 + 3].split(', ')))

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            seg = list(map(int, info[i * 3 + 4].split(', ')))
            if len(seg) <=6 or len(seg) % 2 == 1:
                continue
            anno['category_id'] = label
            anno['bbox'] = bbox
            anno['is_crowd'] = 0
            anno['segmentation'] = [seg]
            anno['bbox_mode'] = BoxMode.XYXY_ABS
            annos.append(anno)
        return annos

    datalist = [line.strip() for line in open(datalist_file).readlines()]
    dataset_dict = []
    name_class = 10000
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        thing_classes = meta.thing_classes
        name_class = len(thing_classes)
        thing_dataset_id_to_contiguous_id = meta.thing_dataset_id_to_contiguous_id
    for i, data in enumerate(datalist):
        record = {}
        record['file_name'] = os.path.join(img_dir, data)
        record['width'], record['height'] = imagesize.get(os.path.join(img_dir, data))
        record['image_id'] = i
        record['annotations'] = _get_annos(os.path.join(label_dir, data[:-4] + '.txt'))
        dataset_dict.append(record)
    return dataset_dict