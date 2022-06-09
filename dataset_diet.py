import json
import shutil
import os

from tqdm import tqdm

import numpy as np
import random

from PIL import Image

from utils.general import xyxy2xywh


def data_prepare():
    random.seed(100)

    path_train_dir = '/DATA/train'
    new_dir = '../dataset_diet'

    # generate dataset_diet/train, dataset_diet/val
    generate_dataset_diet = True
    if generate_dataset_diet == True:
        print('generate dataset_diet/train, dataset_diet/val')

        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir + '/images/train')
        os.makedirs(new_dir + '/images/val')
        os.makedirs(new_dir + '/labels/train')
        os.makedirs(new_dir + '/labels/val')

        with open(path_train_dir + '/label/Train.json') as f:
            json_data = json.load(f)

            # generate images_id_name dict
            json_images = json_data["images"]
            images_id_name = {}

            for image in json_images:
                if image['file_name'].split('_')[1] in ['경기도', '세종특별자치시']:
                    images_id_name[image['id']] = image['file_name']

            trainval_id = images_id_name.keys() # 10065
            val_id = random.sample(trainval_id, 2000)
            train_id = list(set(trainval_id) - set(val_id)) # 10065-2000 = 8065

            json_anno = json_data["annotations"]

            for json_img in tqdm(json_anno):
                image_id = json_img["image_id"]

                if not image_id in images_id_name:
                    continue

                json_img["file_name"] = images_id_name[image_id]

                if image_id in val_id:

                    img_name = json_img['file_name']
                    txt_dir = new_dir + '/labels/val/' + img_name.split('.')[0] + '.txt'
                    img_dir = new_dir + '/images/val/' + img_name

                    f_txt = open(txt_dir, 'a')
                    img_ = Image.open(path_train_dir + '/images/' + img_name)
                    img_size = img_.size

                    # class_id = str(names.index(json_img['category_id']))
                    class_id = json_img['category_id'] - 1
                    img_pos = json_img['bbox'] # xywh

                    # xywh = xyxy2xywh(np.array([[img_pos[0]/img_size[0], img_pos[1]/img_size[1], img_pos[2]/img_size[0], img_pos[3]/img_size[1]]]))[0]
                    x_center = (img_pos[0] + img_pos[2] / 2) / img_size[0]
                    y_center = (img_pos[1] + img_pos[3] / 2) / img_size[1]
                    xywh = np.array([x_center,y_center,img_pos[2]/img_size[0],img_pos[3]/img_size[1]])
                    f_txt.write(f"{class_id} {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n")  # write label

                    f_txt.close()

                    shutil.copy(path_train_dir + '/images/' + img_name, img_dir)

                elif image_id in train_id:

                    img_name = json_img['file_name']
                    txt_dir = new_dir + '/labels/train/' + img_name.split('.')[0] + '.txt'
                    img_dir = new_dir + '/images/train/' + img_name

                    f_txt = open(txt_dir, 'a')
                    img_ = Image.open(path_train_dir + '/images/' + img_name)
                    img_size = img_.size
                    objects_yolo = ''

                    class_id = json_img['category_id'] - 1
                    img_pos = json_img['bbox']

                    x_center = (img_pos[0] + img_pos[2] / 2) / img_size[0]
                    y_center = (img_pos[1] + img_pos[3] / 2) / img_size[1]
                    xywh = np.array([x_center,y_center,img_pos[2]/img_size[0],img_pos[3]/img_size[1]])
                    f_txt.write(f"{class_id} {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n")  # write label

                    f_txt.close()

                    shutil.copy(path_train_dir + '/images/' + img_name, img_dir)


if __name__ == '__main__':
    data_prepare()