import json
import os
import tqdm
import imagesize
import shutil

class FCN2COCO(object):
    def __init__(self, label_root, img_root, coco_root, datalist_file):
        self.label_root = label_root
        self.img_root = img_root
        self.coco_root = coco_root
        self.datalist_file = datalist_file
        self._makedirs()
        self._init_coco()
        self._init_categories()
        self.img_idx = 0
        self.anno_idx = 0

    def _makedirs(self):
        if not os.path.exists(os.path.join(self.coco_root, 'images')):
            os.makedirs(os.path.join(self.coco_root, 'images'))
        if not os.path.exists(os.path.join(self.coco_root, 'annotations')):
            os.makedirs(os.path.join(self.coco_root, 'annotations'))

    def _init_coco(self):
        self._annos = {}
        self._annos['licenses'] = "Charel Create"
        self._annos['images'] = []
        self._annos['annotations'] = []
        self._annos['categories'] = []

    def _init_categories(self):
        categories2id = json.load(open(os.path.join(self.label_root, 'category.json')))
        for name in categories2id:
            category_dict = {}
            category_dict['name'] = name
            category_dict['id'] = categories2id[name]
            self._annos['categories'].append(category_dict)

    def _cp_imgs(self, datalist):
        for data in datalist:
            src_path = os.path.join(self.img_root, data)
            dst_path = os.path.join(self.coco_root, 'images')
            dst_root = os.path.abspath(os.path.dirname(dst_path))
            if not os.path.exists(dst_root):
                os.makedirs(dst_root)
            shutil.copy(src_path, dst_path)

    def _get_annos(self, data):
        ret = []
        label_file = os.path.join(self.label_root, data[:-4] + '.txt')
        label_info = [line.strip() for line in open(label_file, 'r').readlines()]
        num_insts = int(label_info[1])
        for i in range(num_insts):
            anno = {}
            anno['category_id'] = int(label_info[i * 3 + 2])
            bbox = [int(b) for b in label_info[i * 3 + 3].split(', ')]
            anno['bbox'] = [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
            anno['area'] = anno['bbox'][2] * anno['bbox'][3]
            anno['image_id'] = self.img_idx
            anno['id'] = self.anno_idx
            anno['iscrowd'] = 0
            self.anno_idx += 1
            anno['segmentation'] = [[int(p) for p in label_info[i * 3 + 4].split(', ')]]
            ret.append(anno)
        return ret

    def _get_img(self, data):
        ret = {}
        ret['file_name'] = data
        w, h = imagesize.get(os.path.join(self.img_root, data))
        ret['width'] = w
        ret['height'] = h
        ret['id'] = self.img_idx
        self.img_idx += 1
        return ret

    def toCOCO(self):
        datalist = [line.strip() for line in open(self.datalist_file).readlines()]
        self._cp_imgs(datalist)
        for data in tqdm.tqdm(datalist):
            annos = self._get_annos(data)
            if len(annos) == 0:
                continue
            self._annos['annotations'].extend(annos)
            img_dict = self._get_img(data)
            self._annos['images'].append(img_dict)

    def save_coco(self, json_file):
        with open(os.path.join(self.coco_root, 'annotations', json_file), 'w') as fp:
            json.dump(self._annos, fp, separators=(',', ':'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    label_root = '/gruntdata/data/severstal-steel-defect-detection/fcn_label'
    img_root = '/gruntdata/data/severstal-steel-defect-detection/train_images'
    coco_root = '/gruntdata/data/coco/steel'
    datalist_file = '/gruntdata/data/severstal-steel-defect-detection/testlist.txt'
    fcn2coco = FCN2COCO(label_root, img_root, coco_root, datalist_file)
    fcn2coco.toCOCO()
    fcn2coco.save_coco('instances_test.json')
