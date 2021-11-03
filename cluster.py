import sys
import copy
import numpy as np
import logging
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

color_map = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c', '0': '#d62728', '4': '#9467bd', '5': '#8c564b', '6': '#e377c2', '7': '#7f7f7f', '8': '#bcbd22', '9': '#17becf'}

# resize image
def resize(scales, max_size, image_w, image_h):
    short = min(image_w, image_h)
    large = max(image_w, image_h)
    scale = np.random.choice(scales)
    rate = min(scale / short, max_size / large)
    return rate

def whs2xyxy(bboxes):
    ctrs = np.zeros(bboxes.shape)
    x1 = ctrs[:, 0] - bboxes[:, 0] / 2
    y1 = ctrs[:, 1] - bboxes[:, 1] / 2
    x2 = ctrs[:, 0] + bboxes[:, 0] / 2
    y2 = ctrs[:, 1] + bboxes[:, 1] / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def whs_ious(gts, ars):
    gts_d = whs2xyxy(gts)
    ars_d = whs2xyxy(ars)
    num_gts, num_ars = gts_d.shape[0], ars_d.shape[0]
    ars_v = copy.copy(ars_d)[None].repeat(num_gts, axis=0)
    gts_v = copy.copy(gts_d)[:, None].repeat(num_ars, axis=1)
    area_ars = (ars_v[..., 2] - ars_v[..., 0] + 1) * (ars_v[..., 3] - ars_v[..., 1] + 1)
    area_gts = (gts_v[..., 2] - gts_v[..., 0] + 1) * (gts_v[..., 3] - gts_v[..., 1] + 1)

    xmin = np.max(np.stack([ars_v[..., 0], gts_v[..., 0]], axis=-1), axis=-1)
    ymin = np.max(np.stack([ars_v[..., 1], gts_v[..., 1]], axis=-1), axis=-1)
    xmax = np.min(np.stack([ars_v[..., 2], gts_v[..., 2]], axis=-1), axis=-1)
    ymax = np.min(np.stack([ars_v[..., 3], gts_v[..., 3]], axis=-1), axis=-1)
    area_over = (xmax - xmin + 1) * (ymax - ymin + 1)
    ious = area_over / (area_ars + area_gts - area_over)
    return ious

def gt_assignment(anchors, gts):
    ious = whs_ious(gts, anchors)
    dis = 1 - ious
    return np.argmin(dis, axis=0)

def update_ars(assignments, gts, num_ars):
    ars = []
    for idx in range(num_ars):
        match_gts = gts[assignments == idx]
        ar = np.mean(match_gts, axis=0)
        ars.append(ar)
    return np.stack(ars)

def vis(anchors, gts_assign, whs, num_anchors):
    for idx in range(num_anchors):
        gts = whs[gts_assign == idx]
        plt.scatter(gts[:, 0], gts[:, 1], s=1, c=color_map[str(idx + 1)], alpha=0.5)
    plt.scatter(anchors[:, 0], anchors[:, 1], s=5, c=color_map['0'], alpha=1)
    plt.savefig('cluster.png') 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    meta_file = sys.argv[1]
    num_anchors = 5
    coco_loader = COCO(meta_file)
    np.random.seed(1)
    # resize scales and max_size
    scales = [600, 800, 900, 1000]
    max_size = 1000

    meta_imgs = coco_loader.imgs
    annos = coco_loader.imgToAnns
    anno_ids = annos.keys()
    whs = []
    for idx, anno_id in enumerate(anno_ids):
        meta_img = meta_imgs[anno_id]
        bboxes = annos[anno_id]
        # get resize rate of image
        image_w, image_h = meta_img['width'], meta_img['height']
        rate = resize(scales, max_size, image_w, image_h)
        for bbox in bboxes:
            _, _, width, height = bbox['bbox']
            # resize width and height
            width, height = width * rate, height * rate
            whs.append((width, height))
    
    whs = np.stack(whs)
    n_whs = whs.shape[0]
    init_anchor_idx = []
    while len(init_anchor_idx) < num_anchors:
        idx = np.random.randint(n_whs)
        if idx not in init_anchor_idx:
            init_anchor_idx.append(idx)
    
    anchors, diff_assign = whs[init_anchor_idx], 10
    # init gts assignment
    gts_assign = gt_assignment(whs, anchors)
    gts_assign_pre = np.zeros(gts_assign.shape)
    gts_assign_pre.fill(-1)
    
    cluster_time = 0
    while diff_assign > 0:
        cluster_time += 1
        diff_assign = np.sum(np.abs(gts_assign - gts_assign_pre))
        logging.info('{} times cluster | diff with pre {}'.format(cluster_time, diff_assign))
        
        gts_assign_pre = gts_assign
        # update anchors with gts_assign
        anchors = update_ars(gts_assign, whs, num_anchors)
        # update gts assignment
        gts_assign = gt_assignment(whs, anchors)

    vis(anchors, gts_assign, whs, num_anchors)
