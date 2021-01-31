import datasets.coco_custom_buildin
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np
import os

if __name__ == '__main__':
    dataset_name = 'coco_steel_train'
    meta = MetadataCatalog.get(dataset_name)
    print(meta)
    from tqdm import tqdm
    data_dict = DatasetCatalog.get(dataset_name)
    dirname = 'coco-data-vis'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in tqdm(data_dict):
        img = np.array(Image.open(d["file_name"]))
        # for inst in d['annotations']:
            # if inst['category_id'] == 0:
            #     print(d["file_name"])
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
