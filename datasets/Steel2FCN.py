import os
from collections import defaultdict
import pandas as pd
import numpy as np
import imagesize
import cv2
import random
import pycocotools.mask as mask_utils
COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 125, 0)]

class Steel2FCN:
    def __init__(self, data_root, img_dir, label_file, fcn_dir, vis=None):
        self.data_root = data_root
        self.img_dir = img_dir
        self.fcn_dir = fcn_dir
        self.vis = vis
        # if self.vis is not None:
        #     if not os.path.exists()
        if not os.path.exists(os.path.join(data_root, fcn_dir)):
            os.makedirs(os.path.join(data_root, fcn_dir))
        label_info = pd.read_csv(os.path.join(data_root, label_file))
        label_info = label_info[label_info['EncodedPixels'].notnull()]
        self.analysis_bbox = defaultdict(int)
        self.analysis_img = defaultdict(set)
        self.name2rle = defaultdict(list)
        self.name2label = defaultdict(list)
        for i in range(len(label_info)):
            imgId = str(label_info['ImageId'][i])
            self.name2rle[imgId].append(label_info['EncodedPixels'][i])
            self.name2label[imgId].append(label_info['ClassId'][i])
            self.analysis_bbox[label_info['ClassId'][i]] += 1
            self.analysis_img[label_info['ClassId'][i]].add(imgId)
        self.total_list = os.listdir(os.path.join(self.data_root, self.img_dir))
        print("[INFO] Total Images: {}".format(len(self.total_list)))
        print("[INFO] Anno Images: {}".format(len(self.name2rle)))

    def rle2mask(self, rle, imgshape):
        width = imgshape[1]
        height = imgshape[0]
        mask = np.zeros(width * height).astype(np.uint8)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start + lengths[index])] = 1
            current_position += lengths[index]

        return np.flipud(np.rot90(mask.reshape(height, width), k=1))
    def split_train_test(self, ratio = 0.9):
        # train_list = []
        # test_list = []
        datalist = list(self.name2label)
        random.shuffle(datalist)
        train_list = datalist[:int(len(datalist) * ratio)]
        test_list = datalist[int(len(datalist) * ratio) + 1:]
        with open(os.path.join(self.data_root, 'trainlist.txt'), 'w') as fp:
            for t in train_list:
                fp.write(t + '\n')

        with open(os.path.join(self.data_root, 'testlist.txt'), 'w') as fp:
            for t in test_list:
                fp.write(t + '\n')



    def convert2FCN(self):
        print('[INFO] Converting to FCNDataset format...')
        for key, rles in self.name2rle.items():
            labels = self.name2label[key]
            assert len(labels) == len(rles)
            fp = open(os.path.join(self.data_root, self.fcn_dir, key[:-4] + '.txt'), 'w')
            fp_list = []
            fp_list.append(key + '\n')
            # fp_list.append(str(len(labels)) + '\n')
            num_inst = 0
            imgshape = imagesize.get(os.path.join(self.data_root, self.img_dir, key))
            if self.vis is not None:
                org_img = cv2.imread(os.path.join(self.data_root, self.img_dir, key))
                img = org_img.copy()
                full_mask = np.zeros(shape=img.shape, dtype=np.uint8)
            for i in range(len(labels)):
                # fp.write(str(labels[i] + '\n'))
                mask = self.rle2mask(rles[i], imgshape)# .transpose((1, 0))
                # mask = cv2.dilate(mask, np.ones(shape=(3,3)), iterations=5)
                # mask = cv2.blur(mask, (5, 5))
                contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                has_contour = False
                # if len(contours) == 0:
                #     print("[Warning] {}:{} find 0 contours".format(key, i))
                #     continue

                for contour in contours:
                    seg = contour.flatten().tolist()
                    if len(seg) < 8:
                        continue
                    x0, y0, anno_w, anno_h = cv2.boundingRect(contour)
                    if anno_w <= 1 or anno_h <= 1:
                        continue
                    num_inst += 1
                    has_contour = True
                    seg_str = ''
                    for j, point in enumerate(seg):
                        if j != len(seg) - 1:
                            seg_str += str(point) + ', '
                        else:
                            seg_str += str(point) + '\n'
                    bbox_str = '{}, {}, {}, {}\n'.format(x0, y0, x0 + anno_w, y0 + anno_h)
                    fp_list.append(str(labels[i]) + '\n')
                    fp_list.append(bbox_str)
                    fp_list.append(seg_str)
                    if self.vis is not None:
                        # cv2.polyline(img, np.asarray(contour, np.int32), True, COLOR[labels[i] - 1])
                        c_rles = mask_utils.frPyObjects([np.asarray(seg)], imgshape[1], imgshape[0])
                        c_rle = mask_utils.merge(c_rles)
                        segmentation = mask_utils.decode(c_rle)
                        img[segmentation > 0] = COLOR[labels[i] - 1]
                        cv2.rectangle(img, (x0, y0), (x0 + anno_w, y0 + anno_h), thickness=2, color=COLOR[labels[i] - 1])
                        cv2.putText(img, str(labels[i]), (x0, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR[labels[i] - 1], thickness=1)
                    if has_contour == False:
                        print("[Warning] {}:{} find 0 contours".format(key, i))
                        continue
                full_mask[np.where(mask > 0)] = COLOR[labels[i] - 1]
            fp_list.insert(1, str(num_inst) + '\n')
            for line in fp_list:
                fp.write(line)
            fp.close()
            if self.vis is not None:
                save_img = np.zeros(shape=(imgshape[1]* 3, imgshape[0], 3), dtype=np.uint8)
                save_img[0: imgshape[1],...] = org_img
                save_img[imgshape[1]: imgshape[1] * 2, ...] = img
                save_img[imgshape[1] * 2: imgshape[1] * 3, ...] = full_mask
                cv2.imwrite(os.path.join(self.vis, key), save_img)


if __name__ == '__main__':
    data_root = '/media/yons/_gruntdata/data/severstal-steel-defect-detection/'
    img_dir = 'train_images'
    label_file = 'train.csv'
    fcn_dir = 'fcn_label'
    vis = 'vis'
    steel = Steel2FCN(data_root, img_dir, label_file, fcn_dir, vis)
    # steel.convert2FCN()
    steel.split_train_test()


