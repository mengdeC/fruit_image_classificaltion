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
卷积神经网络（CNN）
- **参数选择**：
  - 网络架构：两层卷积层+池化层，接三层全连接层
  - 学习率：0.001
  - 批量大小：32
  - 训练轮数：10
- **实现原理**：
  - CNN通过卷积操作提取图像的局部特征，再通过池化层降低特征维度，最后通过全连接层进行分类。
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np



# 定义超参数
batch_size = 16 # 批处理大小
learning_rate = 0.001 # 学习率
epochs = 10 #训练次数
num_classes = 5  # 类别数量

# 数据预处理：转换为torch张量，并标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化处理
])
# 加载训练数据和测试数据
train_dataset = datasets.ImageFolder(root='dataset_split/test', transform=transform)
test_dataset = datasets.ImageFolder(root='dataset_split/train', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# 定义卷积神经网络模型（使用预训练的ResNet18模型）
class AnimalNet(nn.Module):
    def __init__(self, num_classes):
        super(AnimalNet, self).__init__()
        self.model = models.resnet18(pretrained=True)  # 使用预训练的ResNet18模型
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 修改最后一层

    def forward(self, x):
        x = self.model(x)
        return x


# 创建模型实例并移动到设备
model = AnimalNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

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

# 测试模型并获取预测结果
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