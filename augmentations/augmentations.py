import cv2
import os
import random
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from pycocotools.coco import COCO

def iou_gts_crops(gts, crops):
    area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    xx1 = np.maximum(gts[:, 0], crops[:, 0])
    yy1 = np.maximum(gts[:, 1], crops[:, 1])
    xx2 = np.minimum(gts[:, 2], crops[:, 2])
    yy2 = np.minimum(gts[:, 3], crops[:, 3])
    overlaps = np.stack([xx1, yy1, xx2, yy2], axis=-1)
    area_ovrs = (overlaps[:, 2] - overlaps[:, 0]) * (overlaps[:, 3] - overlaps[:, 1])
    return area_ovrs / area_gts

def Crop(image_path, gts):
    gts = [{'bbox': gt['bbox'], 'category_id': gt['category_id']} for gt in gts]
    image = cv2.imread(image_path)
    cv2.imwrite('initial.jpg', image)
    height, width, _ = image.shape
    logging.info('image height {}, width {}'.format(height, width))
    # crop scale
    scale = random.uniform(0.5, 1)
    short_side = min(width, height)
    w = int(scale * short_side)
    h = w
    logging.info('crop height {}, width {}'.format(h, w))
    left = random.randrange(width - w)
    top = random.randrange(height - h)
    logging.info('crop scale: {}, {}, {}, {}'.format(left, top, left + w, top + h))
    
    # get gts all in crop scale
    gts = np.array([gt['bbox'] for gt in gts])
    gts[:, 2:] += gts[:, :2]
    crops = np.array((left, top, left + w, top + h))
    logging.info('gt_bboxes: {}'.format(gts))
    logging.info('crops: {}'.format(crops))
    # calculate iou of gt self
    ious = iou_gts_crops(gts, crops[np.newaxis])
    logging.info('ious: {}'.format(ious))
    flag = ious >= 1
    bboxes_t = gts[flag]
    logging.info('crop_bboxes: {}'.format(bboxes_t))
   
    rate = min(height / h, width / w) 
    image_t = image[crops[1]:crops[3], crops[0]:crops[2]]
    logging.info('crop_img shape: {}'.format(image_t.shape))
    bboxes_t[:, :2] = np.maximum(bboxes_t[:, :2], crops[:2])
    bboxes_t[:, 2:] = np.minimum(bboxes_t[:, 2:], crops[2:])
    bboxes_t[:, :2] -= crops[:2]
    bboxes_t[:, 2:] -= crops[:2]
    logging.info('relocate: {}'.format(bboxes_t))
    
    # visualize
    image_t = cv2.resize(image_t, None, None, fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)
    for bbox in bboxes_t:
        bbox = list(map(int, bbox * rate))
        cv2.rectangle(image_t, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imwrite('crop.jpg', image_t)
    
def parse(images, meta_file):
    coco_loader = COCO(meta_file)
    Anns = coco_loader.imgToAnns
    meta_imgs = coco_loader.imgs
    annos = {}
    for key in meta_imgs:
        if meta_imgs[key]['file_name'] in images:
            annos[meta_imgs[key]['file_name']] = Anns[key]
    return annos

def main():
    meta_file = '../instances_val2017.json'
    images = 'coco_vals'
    # set random seed
    random.seed(0)
    np.random.seed(0)
    
    root, _, images = list(os.walk(images))[0]
    # get images and annos
    annos = parse(images, meta_file)
    # Crop
    logging.info('--- Crop ---')
    Crop(os.path.join(root, images[0]), annos[images[0]])

if __name__ == "__main__":
    main()
