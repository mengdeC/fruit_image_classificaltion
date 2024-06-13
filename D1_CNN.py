#!/usr/bin/env python38
# -*- coding: UTF-8 -*-
'''
@Project ：图像分类 
@File    ：D1_CNN.py
@IDE     ：PyCharm 
@Author  ：孟德
@Date    ：2024/6/2 15:07 
'''


'''
卷积神经网络的参数选择和实现原理
参数选择
网络架构：使用预训练的ResNet18模型，并在其基础上添加自定义的全连接层。通过冻结预训练模型的参数，仅对新添加的层进行训练。
学习率：0.001，Adam优化器
批量大小：16
训练轮数：30
类别数量：5
数据增强：随机水平翻转、随机旋转、随机垂直翻转等
学习率调度器：StepLR，每3个epoch学习率衰减一半
损失函数：交叉熵损失函数
实现原理
卷积神经网络（CNN）：通过卷积操作提取图像的局部特征，再通过池化层降低特征维度，最后通过全连接层进行分类。
预训练模型：使用在大规模数据集（如ImageNet）上预训练的ResNet18模型，可以利用其学习到的特征来提高小数据集上的性能。通过冻结预训练模型的参数，仅训练新增的全连接层，可以加快训练速度并减少过拟合风险。
数据增强：在训练过程中对图像进行随机变换，可以增加数据的多样性，提高模型的泛化能力。
学习率调度器：1.动态调整学习率，帮助模型更好地收敛。StepLR调度器在每3个epoch后将学习率减半，有助于在训练后期稳定模型的学习。
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
import numpy as np
import pandas as pd

# 定义超参数
batch_size = 16
learning_rate = 0.001
epochs = 30
num_classes = 10

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
# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 定义卷积神经网络模型（使用预训练的ResNet18模型）
class FruitNet(nn.Module):
    def __init__(self, num_classes):
        super(FruitNet, self).__init__()
        self.model = models.resnet18(pretrained=True)  # 使用预训练的ResNet18模型
        # 冻结预训练模型的参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 修改最后的全连接层
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),  # 添加一个线性层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout(0.5),  # Dropout层，防止过拟合
            nn.Linear(512, num_classes)  # 最后的全连接层
        )
    def forward(self, x):
        x = self.model(x) # 输入数据通过模型
        return x


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# 创建模型实例并移动到设备
model = FruitNet(num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 定义学习率调度器
step_scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

# 训练模型
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # 清除之前的梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新所有参数

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # 每训练完一个epoch，更新学习率
    step_scheduler.step()
    # exp_scheduler.step()

    # 清理未使用的变量以释放内存
    del data, target, output
    torch.cuda.empty_cache()

# 在验证集上评估模型
model.eval()  # 设置模型为评估模式
all_preds = []
all_labels = []
with torch.no_grad():  # 在不需要计算梯度的情况下执行，节省内存和计算资源
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

cnn_accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f'CNN Accuracy: {cnn_accuracy:.2f}%')

# 清理未使用的变量以释放内存
del data, target, output
torch.cuda.empty_cache()

# 更新学习率
# lr_scheduler.step(cnn_accuracy)

# 生成分类报告
class_report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# 保存分类报告为CSV文件
report_save_path = 'csv/cnn_classification_report.csv'
class_report_df.to_csv(report_save_path, index=True,header=True)

# 保存模型
# torch.save(model.state_dict(), 'cnn_fruit_classifier.pth')
