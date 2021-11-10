import os
import sys
import json

json_filename = sys.argv[1]
root, _, filenames = list(os.walk('debug_imgs'))[0]
train_json = json.loads(open(json_filename, 'r').readline())

images, annotations, categories = train_json['images'], train_json['annotations'], train_json['categories']
select_imgs, select_annos, id2filename = [], [], {}
for img in images:
    if img['file_name'] in filenames:
        select_imgs.append(img)
        id2filename[img['id']] = img['file_name']
for anno in annotations:
    if anno['image_id'] in id2filename:
        select_annos.append(anno)
select_json = {}
select_json['images'] = select_imgs
select_json['annotations'] = select_annos
select_json['categories'] = categories
print('select_imgs {}'.format(len(select_imgs)))
print('select_annos {}'.format(len(select_annos)))
with open('debug_coco.json', 'w') as f:
    f.write(json.dumps(select_json))