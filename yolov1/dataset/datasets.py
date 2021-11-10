import os
import cv2
import sys
import logging
import torch
from pycocotools.coco import COCO
import torch.utils.data as data

logging.basicConfig(level=logging.INFO)

# COCO dataset
class COCODataset(data.Dataset):
    """
    __getitem__(self, index): output image, gts
    """

    def __init__(self, meta_file, image_dir, root='../../datasets'):
        super(COCODataset, self).__init__()
        # COCO load annotation file
        meta_file = os.path.join(root, meta_file)
        self.meta_file = COCO(meta_file)
        self.image_dir = os.path.join(root, image_dir)
        # get image ids
        self.annos = self.meta_file.imgToAnns
        self.imgs = self.meta_file.imgs
        self.img_ids = sorted(list(self.imgs.keys()))

    # index 
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        meta_img = self.imgs[img_id]
        meta_anno = self.annos[img_id]

        image = cv2.imread(os.path.join(self.image_dir, meta_img['file_name']))
        print(image.shape)
        print(meta_img)
        print(len(meta_anno), meta_anno)

if __name__ == '__main__':
    train_dataset = COCODataset('debug_coco.json', 'debug_imgs')
    logging.info('len annos: {}'.format(len(train_dataset.annos)))
    logging.info('len images: {}'.format(len(train_dataset.imgs)))
    train_dataset.__getitem__(0)
    # logging.info('image_ids: {}'.format(train_dataset.img_ids))