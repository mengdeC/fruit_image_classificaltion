#!/usr/bin/env python38
# -*- coding: UTF-8 -*-
'''
@Project ：图像分类 
@File    ：B清洗图片.py
@IDE     ：PyCharm 
@Author  ：孟德
@Date    ：2024/6/2 14:41 
'''

import os
import cv2
from tqdm import tqdm

dataset_path = 'dataset' # 数据集路径

for animal in tqdm(os.listdir(dataset_path)):
    for file in os.listdir(os.path.join(dataset_path, animal)):
        file_path = os.path.join(dataset_path, animal, file)
        img = cv2.imread(file_path)
        if img is None:
            print(file_path, '读取错误，删除')
            os.remove(file_path)