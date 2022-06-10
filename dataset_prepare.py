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
    new_dir = '../dataset'

    # generate raw_train.json, raw_val.json
    generate_raw_json = False
    if generate_raw_json == True:
        print('generate raw_train.json, raw_val.json')

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
                images_id_name[image['id']] = image['file_name']

            ## 700 + 300
            # trainval_id = random.sample(images_id_name.keys(), 1000)
            # val_id = random.sample(trainval_id, 300)
            # train_id = list(set(trainval_id) - set(val_id)) # 700

            ## 21650 + 3000
            trainval_id = images_id_name.keys() # 24650
            val_id = random.sample(trainval_id, 3000)
            train_id = list(set(trainval_id) - set(val_id)) # 24650-3000 = 21650

            json_anno = json_data["annotations"]

            json_anno_val = []
            json_anno_train = []

            for json_img in tqdm(json_anno):
                image_id = json_img["image_id"]
                json_img["file_name"] = images_id_name[image_id]

                if image_id in val_id:
                    json_anno_val.append(json_img)
                elif image_id in train_id:
                    json_anno_train.append(json_img)

            json_data_val = {}
            json_data_val['annotations'] = json_anno_val
            json_data_train = {}
            json_data_train['annotations'] = json_anno_train

            if os.path.isfile(new_dir + '/raw_val.json'):
                os.remove(new_dir + '/raw_val.json')
            if os.path.isfile(new_dir + '/raw_train.json'):
                os.remove(new_dir + '/raw_train.json')

            with open(new_dir + '/raw_val.json', 'w') as f_val:
                json.dump(json_data_val, f_val)
            with open(new_dir + '/raw_train.json', 'w') as f_train:
                json.dump(json_data_train, f_train)


    # generate dataset/train, dataset/val
    generate_dataset = False
    if generate_dataset == True:
        print('generate dataset/train, dataset/val')

        with open(new_dir + '/raw_val.json') as f:
            json_data = json.load(f)

            json_anno = json_data["annotations"]

            for json_img in tqdm(json_anno):
                img_id = json_img['file_name']
                txt_dir = new_dir + '/labels/val/' + img_id.split('.')[0] + '.txt'
                img_dir = new_dir + '/images/val/' + img_id

                f_txt = open(txt_dir, 'a')
                img_ = Image.open(path_train_dir + '/images/' + img_id)
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

                shutil.copy(path_train_dir + '/images/' + img_id, img_dir)

        with open(new_dir + '/raw_train.json') as f:
            json_data = json.load(f)
            json_anno = json_data["annotations"]

            for json_img in tqdm(json_anno):
                img_id = json_img['file_name']
                txt_dir = new_dir + '/labels/train/' + img_id.split('.')[0] + '.txt'
                img_dir = new_dir + '/images/train/' + img_id

                f_txt = open(txt_dir, 'a')
                img_ = Image.open(path_train_dir + '/images/' + img_id)
                img_size = img_.size
                objects_yolo = ''

                class_id = json_img['category_id'] - 1
                img_pos = json_img['bbox']

                x_center = (img_pos[0] + img_pos[2] / 2) / img_size[0]
                y_center = (img_pos[1] + img_pos[3] / 2) / img_size[1]
                xywh = np.array([x_center,y_center,img_pos[2]/img_size[0],img_pos[3]/img_size[1]])
                f_txt.write(f"{class_id} {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n")  # write label

                f_txt.close()

                shutil.copy(path_train_dir + '/images/' + img_id, img_dir)


    # generate dataset/test_imgs.txt
    generate_dataset_test_imgs = True
    if generate_dataset_test_imgs == True:
        f_txt = open('../dataset/test_imgs.txt', 'w')

        with open('/DATA/test/Test_Images_Information.json') as f:
            json_data = json.load(f)

            for image in tqdm(json_data["images"]):
                image_path = "/DATA/test/images/" + image["file_name"]
                f_txt.write(f"{image_path}\n")


    # generate dataset_diet/train
    generate_dataset_diet = True
    if generate_dataset_diet == True:
        print('generate dataset_diet/train')

        diet_dir = '../dataset_diet'
        if os.path.exists(diet_dir):
            shutil.rmtree(diet_dir)
        os.makedirs(diet_dir + '/images/train')
        os.makedirs(diet_dir + '/labels/train')

        with open(path_train_dir + '/label/Train.json') as f:
            json_data = json.load(f)

            # generate images_id_name dict
            json_images = json_data["images"]
            images_id_name = {}

            for image in json_images:
                images_id_name[image['id']] = image['file_name']

            # count images per class
            count_classes = [0] * 14
            json_anno = json_data["annotations"]
            for anno in json_anno:
                count_classes[anno["category_id"]-1] += 1
            print('Count images per classes: ', count_classes)
            print('Ratio: ', [round(100*i/sum(count_classes),4) for i in count_classes])
            print('Total images: ', len(images_id_name))

            # generate image_id_diet
            print("!!!!!DIET!!!!!")
            image_id_diet = []
            json_anno = json_data["annotations"]
            i_12 = 0
            for anno in json_anno:
                if anno["category_id"] in [4, 5, 7, 9, 11, 13]:
                    if not anno["image_id"] in image_id_diet:
                        image_id_diet.append(anno["image_id"])
                if anno["category_id"] in [12]:
                    i_12 += 1
                    if i_12 % 20 == 0 and not anno["image_id"] in image_id_diet:
                        image_id_diet.append(anno["image_id"])

            # count images per class - diet
            count_classes_diet = [0] * 14
            for anno in tqdm(json_anno):
                if anno["image_id"] in image_id_diet:
                    count_classes_diet[anno["category_id"]-1] += 1
            print('count images per classs: ', count_classes_diet)
            print('Ratio: ', [round(100*i/sum(count_classes_diet),4) for i in count_classes_diet])
            print('Total images: ', len(image_id_diet))

            # set dataset_diet
            new_diet_dir = '../dataset_diet'
            if os.path.exists(new_diet_dir):
                shutil.rmtree(new_diet_dir)
            os.makedirs(new_diet_dir + '/images/train')
            os.makedirs(new_diet_dir + '/labels/train')

            # set images&labels in dataset_diet
            for anno in tqdm(json_anno):
                if anno["image_id"] in image_id_diet:
                    img_id = images_id_name[anno["image_id"]]
                    txt_dir = new_diet_dir + '/labels/train/' + img_id.split('.')[0] + '.txt'
                    img_dir = new_diet_dir + '/images/train/' + img_id

                    f_txt = open(txt_dir, 'a')
                    img_ = Image.open(path_train_dir + '/images/' + img_id)
                    img_size = img_.size
                    objects_yolo = ''

                    class_id = anno['category_id'] - 1
                    img_pos = anno['bbox']

                    x_center = (img_pos[0] + img_pos[2] / 2) / img_size[0]
                    y_center = (img_pos[1] + img_pos[3] / 2) / img_size[1]
                    xywh = np.array([x_center,y_center,img_pos[2]/img_size[0],img_pos[3]/img_size[1]])
                    f_txt.write(f"{class_id} {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n")  # write label

                    f_txt.close()

                    shutil.copy(path_train_dir + '/images/' + img_id, img_dir)


if __name__ == '__main__':
    data_prepare()