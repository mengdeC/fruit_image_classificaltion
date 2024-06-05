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

# 定义超参数
batch_size = 16
learning_rate = 0.001
epochs = 50
num_classes = 10

# 数据预处理：转换为torch张量，并标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomAffine(10),  # 随机仿射变换
    transforms.GaussianBlur(kernel_size=3),  # 高斯模糊
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化处理
])

# 加载训练数据和测试数据
train_dataset = datasets.ImageFolder(root='fruitdatasets_split/train', transform=transform)
test_dataset = datasets.ImageFolder(root='fruitdatasets_split/test', transform=transform)
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
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 定义学习率调度器 88%
# lr_scheduler：学习率调度器
# ReduceLROnPlateau：减少学习率调度器
# 参数optimizer：优化器
# mode='max'：模式，以最大化指标为准
# factor=0.5：因子，以0.5为步长下降
# patience=2：容忍度，当指标不再改善的次数
# verbose=True：是否打印信息
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# 定义步长调度器 85%
# step_scheduler：步长调度器
# StepLR：步长调度器
# 参数optimizer：优化器
# step_size=3：步长，以3为步长
# gamma=0.5：因子，以0.5为步长下降
step_scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

# 定义指数调度器 86%
# exp_scheduler：指数调度器
# ExponentialLR：指数调度器
# 参数optimizer：优化器
# gamma=0.95：因子，以0.95为指数下降
exp_scheduler = ExponentialLR(optimizer, gamma=0.95)

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

# 计算分类指标
print("CNN Classification Report")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# 更新学习率
lr_scheduler.step(cnn_accuracy)
# step_scheduler.step()
# exp_scheduler.step()

# 保存模型
# torch.save(model.state_dict(), 'cnn_fruit_classifier.pth')
