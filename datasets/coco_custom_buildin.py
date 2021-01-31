import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from .fcn import get_fcn_dict
# from .buildin_meta import _get_steel_fcn_instances_meta, _get_wheat_fcn_instances_meta
from detectron2.data.datasets import register_coco_instances
from .coco_custom_builtin_meta import _get_coco_steel_instances_meta

_STEEL_COCO = {
    "coco_steel_train": ("steel/images",
                        "steel/annotations/instances_train.json",
                        ),

    "coco_steel_test": ("steel/images",
                        "steel/annotations/instances_test.json",
                        ),
}

# _WHEAT_FCN = {
#     "fcn_wheat_train": ("WheatDetection/train",
#                         "WheatDetection/fcn_label",
#                         'WheatDetection/trainlist.txt'),
#
#     "fcn_wheat_test": ("WheatDetection/train",
#                        "WheatDetection/fcn_label",
#                        'WheatDetection/testlist.txt'),
# }



def register_all_coco(root):
    for name, name_info in _STEEL_COCO.items():
        register_coco_instances(
            name,
            _get_coco_steel_instances_meta(),
            json_file=os.path.join(root, _STEEL_COCO[name][1]),
            image_root=os.path.join(root, _STEEL_COCO[name][0]),
        )

    # for name, name_info in _WHEAT_FCN.items():
    #     register_fcn_instances(
    #         name,
    #         _get_wheat_fcn_instances_meta(),
    #         label_dir=os.path.join(root, _WHEAT_FCN[name][1]),
    #         image_dir=os.path.join(root, _WHEAT_FCN[name][0]),
    #         datalist_file=os.path.join(root, _WHEAT_FCN[name][2])
    #     )



_root = '/gruntdata/data/coco'

register_all_coco(_root)
