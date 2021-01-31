import os
import cv2
import json
import random
from collections import defaultdict
import pandas as pd

class Wheat2FCN(object):
    def __init__(self, data_root, csv_file):
        csv_df = pd.read_csv(os.path.join(data_root, csv_file))
        self.csv_df = csv_df
        self.data_root = data_root
        self.datalist = csv_df.image_id.unique()
        print("[INFO] Total Images: {}".format(len(self.datalist)))
        print("[INFO] Total Annos: {}".format(len(csv_df.image_id)))
        # self.imgId2Annos = {}
        self.imgId2Annos = defaultdict(list)
        for i, img_id in enumerate(self.csv_df.image_id):
            self.imgId2Annos[img_id].append(self.csv_df.bbox[i])

    def convert2FCN(self, fcn_label_dir='fcn_label', vis=False):
        fcn_label_path = os.path.join(self.data_root, fcn_label_dir)
        if not os.path.exists(fcn_label_path):
            os.makedirs(fcn_label_path)

        for img_id in self.datalist:
            if vis:
                img = cv2.imread(os.path.join(self.data_root, 'train', img_id + '.jpg'))
                if not os.path.exists(os.path.join(self.data_root, 'vis')):
                    os.makedirs(os.path.join(self.data_root, 'vis'))
            with open(os.path.join(fcn_label_path, img_id + '.txt'), 'w') as fp:
                fp.write(img_id + '\n')
                fp.write(str(len(self.imgId2Annos[img_id])) + '\n')
                for bbox in self.imgId2Annos[img_id]:
                    fp.write(str(1) + '\n')
                    bbox = json.loads(bbox)
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                    segmentation = [x1, y1, x2, y1, x1, y2, x2, y2]
                    fp.write('{}, {}, {}, {}\n'.format(x1, y1,
                                                       x2, y2))

                    for i, p in enumerate(segmentation):
                        if i == len(segmentation) - 1:
                            fp.write(str(p) + '\n')
                        else:
                            fp.write(str(p) + ', ')

                    if vis:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                if vis:
                    cv2.imwrite(os.path.join(self.data_root, 'vis', img_id + '.jpg'), img)
    def splitTrainTest(self, train_ratio=0.9):
        num_images = len(self.datalist)
        random.shuffle(self.datalist)
        num_train = int(num_images * train_ratio)
        with open(os.path.join(self.data_root, 'trainlist.txt'), 'w') as fp:
            for img_id in self.datalist[:num_train]:
                fp.write(img_id + '.jpg\n')

        with open(os.path.join(self.data_root, 'testlist.txt'), 'w') as fp:
            for img_id in self.datalist[num_train:]:
                fp.write(img_id + '.jpg\n')


if __name__ == '__main__':
    data_root = '/media/yons/gruntdata/data/WheatDetection'
    csv_file = 'train.csv'
    parser = Wheat2FCN(data_root, csv_file)
    parser.convert2FCN(vis=False)
    parser.splitTrainTest()






