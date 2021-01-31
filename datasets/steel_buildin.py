import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from .fcn import get_fcn_dict
from .buildin_meta import _get_steel_fcn_instances_meta, _get_wheat_fcn_instances_meta

_STEEL_FCN = {
    "fcn_steel_train": ("severstal-steel-defect-detection/train_images",
                        "severstal-steel-defect-detection/fcn_label",
                        'severstal-steel-defect-detection/trainlist.txt'),

    "fcn_steel_test": ("severstal-steel-defect-detection/train_images",
                       "severstal-steel-defect-detection/fcn_label",
                       'severstal-steel-defect-detection/testlist.txt'),
}

_WHEAT_FCN = {
    "fcn_wheat_train": ("WheatDetection/train",
                        "WheatDetection/fcn_label",
                        'WheatDetection/trainlist.txt'),

    "fcn_wheat_test": ("WheatDetection/train",
                       "WheatDetection/fcn_label",
                       'WheatDetection/testlist.txt'),
}

def register_fcn_instances(dataset_name, meta, label_dir, image_dir, datalist_file):
    DatasetCatalog.register(dataset_name, lambda:
                        get_fcn_dict(image_dir, label_dir, datalist_file, dataset_name))

    MetadataCatalog.get(dataset_name).set(label_dir=label_dir, image_dir=image_dir, **meta)


def register_all_fcn(root):
    for name, name_info in _STEEL_FCN.items():
        register_fcn_instances(
            name,
            _get_steel_fcn_instances_meta(),
            label_dir=os.path.join(root, _STEEL_FCN[name][1]),
            image_dir=os.path.join(root, _STEEL_FCN[name][0]),
            datalist_file=os.path.join(root, _STEEL_FCN[name][2])
        )

    for name, name_info in _WHEAT_FCN.items():
        register_fcn_instances(
            name,
            _get_wheat_fcn_instances_meta(),
            label_dir=os.path.join(root, _WHEAT_FCN[name][1]),
            image_dir=os.path.join(root, _WHEAT_FCN[name][0]),
            datalist_file=os.path.join(root, _WHEAT_FCN[name][2])
        )



_root = '/gruntdata/data'

register_all_fcn(_root)
