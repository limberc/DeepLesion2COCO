"""
1. read csv，save as cocostyle dataset：
--dataset/
  --trainset/
    --image/
      --***.jpg
        ***.jpg
        ...
    --annotation/
      --annotation.json
  --validset/
  --testset/
"""
import json
import os
import pdb
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from load_ct_img import load_prep_img

label_dict = {}
label_dict[1] = 'all_type'
'''
label_dict[1]  = 'bone'
label_dict[2]  = 'abdomen'
label_dict[3]  = 'mediastinum'
label_dict[4] = 'liver'
label_dict[5] = 'lung'
label_dict[6]  = 'kidney'
label_dict[7] = 'soft tissue'
label_dict[8] = 'pelvis'
'''

im_file_path = './data/Images_png/'
anns_path = './data/DL_info.csv'
output_root = './data/deeplesion_cocostyle/'
output_path = []
output_path.append(os.path.join(output_root, 'train'))
output_path.append(os.path.join(output_root, 'valid'))
output_path.append(os.path.join(output_root, 'test'))


def transform_deeplesion2coco(anns_all):
    '''
    input:
    anns_all:  the annotation provided (Dataframe).
    '''

    # 初始化dataset
    Dataset = list()  # Dataset[0]为training set  1为valid set 2为test set
    for i in range(3):
        dataset = dict()
        dataset['images'] = []
        dataset['type'] = 'instances'
        dataset['annotations'] = []
        dataset['categories'] = []
        dataset['info'] = None
        dataset['licenses'] = None
        Dataset.append(dataset)
    annotation_id = [0, 0, 0]
    image_id = [0, 0, 0]

    # add dataset['categories']
    for category_id, category_name in label_dict.items():
        category_item = dict()
        category_item['supercategory'] = category_name
        category_item['id'] = category_id
        category_item['name'] = category_name
        for dataset in Dataset:
            dataset['categories'].append(category_item)

    for path in output_path:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(os.path.join(path, 'image'))
            os.makedirs(os.path.join(path, 'annotation'))
        else:
            os.makedirs(path)
            os.makedirs(os.path.join(path, 'image'))
            os.makedirs(os.path.join(path, 'annotation'))

    # using a list to choose the tail index of repeated annotations of the same image
    # assume that multi-lesion of the same image has continuous index
    multi_lesion_index = []
    file_name = None
    for index, row in anns_all.iterrows():
        if file_name == row['File_name']:
            multi_lesion_index.append(index)
        file_name = row['File_name']

    for index, row in tqdm(anns_all.iterrows()):
        file_name = row['File_name']
        datatype = row.Train_Val_Test - 1
        img_path = ''.join([_ + '_' for _ in row['File_name'].split('_')[:-1]])[:-1] + '/' + \
                   row['File_name'].split('_')[-1]
        spacing3D = np.array([list(map(float, row['Spacing_mm_px_'].split(",")))])
        spacing = spacing3D[:, 0]
        slice_intv = spacing3D[:, 2]
        is_train = True if datatype == 0 else False
        img, _, _ = load_prep_img(im_file_path, img_path,
                                  spacing, slice_intv, is_train=is_train)
        cv2.imwrite(os.path.join(output_path[datatype], 'image', file_name[0:-3] + 'jpg'), img)
        # add 'image'
        if index not in multi_lesion_index:
            image = dict()
        image['id'] = image_id[datatype]
        image_id[datatype] = image_id[datatype] + 1
        image['file_name'] = file_name[0:-3] + 'jpg'
        image['width'] = img.shape[1]
        image['height'] = img.shape[0]
        Dataset[datatype]['images'].append(image)
        # add 'annotations'
        annotation_item = dict()
        bbox = [float(i) for i in row.Bounding_boxes.split(',')]
        x1 = min(bbox[0], bbox[2])
        y1 = min(bbox[1], bbox[3])
        x2 = max(bbox[0], bbox[2])
        y2 = max(bbox[1], bbox[3])
        x = x1
        y = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        annotation_item['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        annotation_item['image_id'] = image['id']
        annotation_item['iscrowd'] = 0
        annotation_item['bbox'] = [x, y, w, h]
        annotation_item['area'] = w * h
        annotation_item['id'] = annotation_id[datatype]
        annotation_id[datatype] = annotation_id[datatype] + 1
        annotation_item['category_id'] = 1
        annotation_item['spacing'] = spacing
        annotation_item['slice_intv'] = slice_intv

        Dataset[datatype]['annotations'].append(annotation_item)
    pdb.set_trace()
    for i in range(3):
        json.dump(Dataset[i],
                  open(os.path.join(output_path[i], 'annotation', 'annotation.json'), 'w'))


if __name__ == '__main__':
    anns_all = pd.read_csv(anns_path)
    transform_deeplesion2coco(anns_all)
