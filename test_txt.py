import json
import shutil
import os

from tqdm import tqdm

import numpy as np
import random

from PIL import Image

from utils.general import xyxy2xywh


def test_txt():

    f_txt = open('../dataset/test_imgs.txt', 'w')

    with open('/DATA/test/Test_Images_Information.json') as f:
        json_data = json.load(f)

        for image in tqdm(json_data["images"]):
            image_path = "/DATA/test/images/" + image["file_name"]
            f_txt.write(f"{image_path}\n")

if __name__ == '__main__':
    test_txt()