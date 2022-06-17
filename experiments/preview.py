import os

import random
import shutil

def img_random_copy(task, num):
    path = "/DATA/" + task + "/images/"
    img_list = os.listdir(path)

    preview_path = "preview/" + task + "/"
    if os.path.exists(preview_path):
        shutil.rmtree(preview_path)
    os.makedirs(preview_path)

    for img_name in random.sample(img_list, num):
        shutil.copyfile(path + img_name, preview_path + img_name)

if False:
    img_random_copy("train", 100)
    img_random_copy("test", 100)

def train_city_count():
    city_dict = {}

    path = "/DATA/train/images/"
    img_list = os.listdir(path)

    for img_name in img_list:
        city_name = img_name.split("_")[1]
        if city_name in city_dict.keys():
            city_dict[city_name] += 1
        else:
            city_dict[city_name] = 1
    
    print(city_dict)

train_city_count()