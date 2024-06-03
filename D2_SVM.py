#!/usr/bin/env python38
# -*- coding: UTF-8 -*-
'''
@Project ：图像分类 
@File    ：D2_SVM.py
@IDE     ：PyCharm 
@Author  ：孟德
@Date    ：2024/6/2 15:19 
'''

'''
支持向量机（SVM）
- **参数选择**：
  - 核函数：径向基函数（RBF）
  - 正则化参数：C=1
- **实现原理**：
  - SVM通过在高维空间中找到一个最优超平面来分隔不同类别的样本，使得边界两侧的间隔最大化。
'''

import torch
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# 定义超参数
batch_size = 16 # 批量大小
num_classes = 10 # 类别数

# 数据预处理：转换为torch张量，并标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化处理
])

# 加载训练数据和测试数据
train_dataset = datasets.ImageFolder(root='dataset_split/test', transform=transform)
test_dataset = datasets.ImageFolder(root='dataset_split/train', transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 创建 ResNet18 模型
resnet18 = models.resnet18(pretrained=True)
resnet18 = resnet18.to(device)

# 提取训练数据特征
train_features = []
train_labels = []

# 遍历训练数据集
for data, target in train_loader:
    data = data.to(device)  # 移动数据到相同的设备上
    with torch.no_grad():  # 在不需要计算梯度的情况下执行，节省内存和计算资源
        features = resnet18(data).cpu().numpy()
    train_features.extend(features)
    train_labels.extend(target.numpy())

# 提取测试数据特征
test_features = []
test_labels = []

# 遍历测试数据集
for data, target in test_loader:
    data = data.to(device)  # 移动数据到相同的设备上
    with torch.no_grad():  # 在不需要计算梯度的情况下执行，节省内存和计算资源
        features = resnet18(data).cpu().numpy()
    test_features.extend(features)
    test_labels.extend(target.numpy())

# 将特征和标签转换为numpy数组
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# 定义和训练SVM分类器
svm_clf = svm.SVC(kernel='linear') # 使用线性核
svm_clf.fit(train_features, train_labels) # 使用训练数据进行训练

# 测试SVM分类器
svm_predictions = svm_clf.predict(test_features) # 使用测试数据进行预测
svm_accuracy = accuracy_score(test_labels, svm_predictions) # 计算准确率

print(f'SVM Accuracy: {svm_accuracy:.2f}%')

# 计算分类指标
print("SVM Classification Report")
print(classification_report(test_labels, svm_predictions, target_names=train_dataset.classes))
