
COCO_STEEL_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "1"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "2"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "3"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "4"},]

def _get_coco_steel_instances_meta():
    thing_ids = [k["id"] for k in COCO_STEEL_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_STEEL_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 4, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_STEEL_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret