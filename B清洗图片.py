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
import shutil  # 导入shutil模块用于删除文件夹


def clean_and_preprocess_images(dataset_path, output_path):
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历数据集中的所有水果类别
    for fruit in tqdm(os.listdir(dataset_path)):
        fruit_path = os.path.join(dataset_path, fruit)  # 获取水果类别的路径
        if os.path.isdir(fruit_path):  # 确保它是一个目录
            output_fruit_path = os.path.join(output_path, fruit)  # 获取输出目录的路径
            if not os.path.exists(output_fruit_path):
                os.makedirs(output_fruit_path)

            # 遍历该水果类别中的所有图像文件
            for file in os.listdir(fruit_path):
                file_path = os.path.join(fruit_path, file)  # 获取文件的完整路径
                img = cv2.imread(file_path)  # 读取图像文件

                # 检查图像是否正确读取
                if img is None:
                    print(f"{file_path} 读取错误，跳过处理")
                else:
                    # 转换图像格式和尺寸

                    output_img = cv2.resize(img, (224, 224))  # 调整尺寸为224x224

                    # 保存转换和调整后的图像文件
                    output_file_path = os.path.join(output_fruit_path, file)
                    cv2.imwrite(output_file_path, output_img)  # 保存图像

    # 确保输出文件夹中存在文件，避免删除空文件夹
    if os.listdir(output_path):
        # 删除原始的dataset文件夹
        shutil.rmtree(dataset_path)  # 使用shutil.rmtree来删除整个文件夹
        print(f"{dataset_path} 已被删除")


# 数据集路径
dataset_path = "dataset"

# 输出数据集路径
output_dataset_path = "fruitdatasets"

if __name__ == '__main__':
    clean_and_preprocess_images(dataset_path, output_dataset_path)