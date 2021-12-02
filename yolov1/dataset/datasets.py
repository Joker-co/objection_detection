import os
import cv2
import sys
import logging
import torch
import copy
import numpy as np
from pycocotools.coco import COCO
import torch.utils.data as data

from yolov1.dataset.transforms import RandomFlip, RandomResize, ToTensor, Normalize

logging.basicConfig(level=logging.INFO)

# COCO dataset
class COCODataset(data.Dataset):
    """
    __getitem__(self, index): output image, gts
    """

    def __init__(self, meta_file, image_dir, scales=[416], max_scale=416,
                 means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        super(COCODataset, self).__init__()
        # COCO load annotation file
        self.meta_file = COCO(meta_file)
        self.image_dir = image_dir
        # get image ids
        self.annos = self.meta_file.imgToAnns
        self.imgs = self.meta_file.imgs
        self.img_ids = sorted(list(self.imgs.keys()))
        # categories
        self.cats = self.meta_file.cats
        # process continue cats
        self.cat_even2continue = {
            c_id: idx + 1 for idx, c_id in enumerate(sorted(self.cats.keys()))
        }

        # transforms
        # random flip
        self.random_flip = RandomFlip()
        # Resize
        self.resize = RandomResize(scales, max_scale)
        # ToTensor
        self.totensor = ToTensor()
        # Normalize
        self.normalize = Normalize(means, stds)

    def __len__(self):
        return len(self.img_ids)

    def transform(self, image, gt_bboxes):
        image, gt_bboxes = copy.copy(image), copy.copy(gt_bboxes)
        # random flip
        image, gts = self.random_flip(image, gt_bboxes)
        # Resize
        image, gts, scale_factor = self.resize(image, gts)
        resize_h, resize_w = image.shape[:2]
        # ToTensor
        image, gts = self.totensor(image, gts)
        # normalize
        image = self.normalize(image)
        return image, gts, scale_factor, resize_h, resize_w

    def vis(self, image, bboxes, save_path):
        image, bboxes = copy.copy(image), copy.copy(bboxes)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for bbox in bboxes:
            x1, y1, x2, y2, label = list(map(int, bbox))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image = cv2.putText(image, self.cats[label]['name'], (x1, y1 - 3), font, 0.3, (255, 255, 255), 1)
        cv2.imwrite(save_path, image)

    # index 
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        meta_img = self.imgs[img_id]
        meta_anno = self.annos[img_id]

        # BGR
        image = cv2.imread(os.path.join(self.image_dir, meta_img['file_name']))
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_bboxes = []
        for anno in meta_anno:
            # x1, y1, w, h
            bbox = anno['bbox']
            cat = self.cat_even2continue[anno['category_id']]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            gt_bboxes.append([x1, y1, x2, y2, cat])
        # gt_bboxes -> numpy.array
        gt_bboxes = np.array(gt_bboxes)

        image, gt_bboxes, scale_factor, resize_h, resize_w = self.transform(image, gt_bboxes)
        # record scale_factor
        # [h, w]
        image_info = [[scale_factor, scale_factor], resize_h, resize_w]

        # dict
        output = {}
        output['image'] = image
        output['gt_bboxes'] = gt_bboxes
        output['image_info'] = image_info
        return output

if __name__ == '__main__':
    train_dataset = COCODataset('debug_coco.json', 'debug_imgs')
    logging.info('len annos: {}'.format(len(train_dataset.annos)))
    logging.info('len images: {}'.format(len(train_dataset.imgs)))
    logging.info('cats: {}'.format(train_dataset.cats))
    train_dataset.__getitem__(0)
    logging.info('image_ids: {}'.format(train_dataset.img_ids))