import sys
import json
import logging

logging.basicConfig(level=logging.INFO)

custom_file_name = sys.argv[1]
custom_file = open(custom_file_name, 'r').readlines()
logging.info('Load custom file.')

# coco format
# coco_file = 'instances_val2017.json'
# coco_file = json.loads(open(coco_file, 'r').readline())
# ['info', 'licenses', 'images', 'annotations', 'categories']
# image {'file_name': '000000397133.jpg', 'height': 427, 'width': 640, 'id': 397133}
# annotations {'area': 702.1057499999998, 'iscrowd': 0, 'image_id': 289343, 
# 'bbox': [473.07, 395.93, 38.65, 28.67], 'category_id': 18, 'id': 1768}
# categories [{'supercategory': 'person', 'id': 1, 'name': 'person'}, 
# {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}]

logging.info('Start converting.')
coco_imgs = []
coco_annos = []
coco_cats = []
cats_set = set()
img_id, anno_id = 0, 0
# {"filename": "000001.jpg", "image_height": 500, "image_width": 353, 
# "instances": 
# [{"is_ignored": false, "bbox": [47.0, 239.0, 194.0, 370.0], "label": 12}, {"is_ignored": false, "bbox": [7.0, 11.0, 351.0, 497.0], "label": 15}]}
for anno in custom_file:
    custom_anno = json.loads(anno.strip())
    conv_img = {}
    img_id += 1
    conv_img['file_name'] = custom_anno['filename']
    conv_img['height'] = custom_anno['image_height']
    conv_img['width'] = custom_anno['image_width']
    conv_img['id'] = img_id
    coco_imgs.append(conv_img)
    custom_instances = custom_anno['instances']
    for cus_ins in custom_instances:
        anno_id += 1
        conv_anno = {}
        cat = cus_ins['label']
        x1, y1, x2, y2 = cus_ins['bbox']
        w, h = (x2 - x1 + 1), (y2 - y1 + 1)
        area = w * h
        conv_anno['area'] = area
        # conv_anno['iscrowd'] = 0 if not cus_ins['is_ignored'] else 1
        conv_anno['image_id'] = img_id
        conv_anno['bbox'] = [x1, y1, w, h]
        conv_anno['category_id'] = cat
        conv_anno['id'] = anno_id
        coco_annos.append(conv_anno)
        cats_set.add(cat)
# categories
CLASS_NAMES = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
cats_set = list(cats_set)
for cat_id in cats_set:
    cat = {'id': cat_id, 'name': CLASS_NAMES[cat_id]}
    coco_cats.append(cat)

logging.info('Dump converting results.')
coco_dict = {'images': coco_imgs, 'annotations': coco_annos, 'categories': coco_cats}
with open(custom_file_name.split('.')[0] + '_2coco.json', 'w') as f:
    f.write(json.dumps(coco_dict))