#!/usr/bin/env python38
# -*- coding: UTF-8 -*-
'''
@Project ：图像分类 
@File    ：D3_KNN.py
@IDE     ：PyCharm 
@Author  ：孟德
@Date    ：2024/6/2 15:33 
'''

'''
K近邻算法（KNN）
- **参数选择**：
  - K值选择：5
  - 距离度量：欧氏距离
- **实现原理**：
  - 对于新图像，计算它与所有训练样本的距离，选取距离最近的K个样本，根据这K个样本的标签进行投票，确定新图像的类别。
'''

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# 定义超参数
batch_size = 16 # 批量大小
num_classes = 10 # 类别数

# 数据预处理
# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224), # 随机裁剪
                                      transforms.RandomHorizontalFlip(), # 随机水平翻转
                                      transforms.ToTensor(), # 转 Tensor
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
                                     ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256), # 缩放
                                     transforms.CenterCrop(224), # 中心裁剪
                                     transforms.ToTensor(), # 转 Tensor
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) # 归一化
                                    ])

# 加载训练数据和测试数据
train_dataset = datasets.ImageFolder(root='fruitdatasets_split/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='fruitdatasets_split/test', transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 创建 ResNet18 模型
resnet18 = models.resnet18(pretrained=True)
resnet18 = resnet18.to(device)

# 提取训练数据特征
train_features = []
train_labels = []

for data, target in train_loader:
    data = data.to(device)  # 移动数据到相同的设备上
    with torch.no_grad():  # 在不需要计算梯度的情况下执行，节省内存和计算资源
        features = resnet18(data).cpu().numpy()
    train_features.extend(features)
    train_labels.extend(target.numpy())

# 提取测试数据特征
test_features = []
test_labels = []

for data, target in test_loader:
    data = data.to(device)  # 移动数据到相同的设备上
    with torch.no_grad():  # 在不需要计算梯度的情况下执行，节省内存和计算资源
        features = resnet18(data).cpu().numpy()
    test_features.extend(features)
    test_labels.extend(target.numpy())

# 将特征和标签转换为NumPy数组
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# 定义和训练KNN分类器
knn_clf = KNeighborsClassifier(n_neighbors=5) # 选择K=5
knn_clf.fit(train_features, train_labels) # 使用训练数据进行训练

# 测试KNN分类器
knn_predictions = knn_clf.predict(test_features) # 使用测试数据进行预测
knn_accuracy = accuracy_score(test_labels, knn_predictions) # 计算准确率

print(f'KNN Accuracy: {knn_accuracy:.2f}%')

# 保存评估结果到文件
accuracy_save_path = 'csv/knn_accuracy.csv'
with open(accuracy_save_path, 'w') as f:
    f.write(classification_report(test_labels, knn_predictions, target_names=train_dataset.classes))

# 生成分类报告
class_report = classification_report(test_labels, knn_predictions, target_names=train_dataset.classes, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# 保存分类报告为CSV文件
report_save_path = 'csv/knn_classification_report.csv'
class_report_df.to_csv(report_save_path, index=True,header=True)