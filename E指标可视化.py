#!/usr/bin/env python38
# -*- coding: UTF-8 -*-
'''
@Project ：图像分类 
@File    ：E指标可视化.py
@IDE     ：PyCharm 
@Author  ：孟德
@Date    ：2024/6/13 22:43 
'''

#!/usr/bin/env python38
# -*- coding: UTF-8 -*-
'''
@Project ：图像分类 
@File    ：E指标可视化.py
@IDE     ：PyCharm 
@Author  ：孟德
@Date    ：2024/6/13 22:43 
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def visualize_classification_report(csv_file_path, save_path=None):
    # 读取分类报告的CSV文件
    df = pd.read_csv(csv_file_path)

    # 去掉不需要的行（总计行）
    df = df[:-3]  # 去掉最后三行：accuracy, macro avg, weighted avg

    # 提取标签、精确率、召回率和F1-score
    labels = df['类别']
    precisions = df['精确率']
    recalls = df['召回率']
    f1_scores = df['F1-score']

    # 设置图表参数
    x = np.arange(len(labels))  # 标签的位置
    width = 0.2  # 柱状图的宽度

    # 创建子图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制柱状图
    rects1 = ax.bar(x - width, precisions, width, label='Precision')
    rects2 = ax.bar(x, recalls, width, label='Recall')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-score')

    # 添加标签、标题和自定义x轴刻度
    ax.set_ylabel('Scores')
    ax.set_title('Classification Report')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    # 添加数值标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    # 调整布局
    fig.tight_layout()

    # 显示图表或保存图表
    if save_path:
        fig.savefig(save_path)
        print(f"保存成功: {save_path}")
    else:
        plt.show()

if __name__ == '__main__':

    # 示例使用方法
    csv_file_path = 'csv/cnn_classification_report.csv'
    visualize_classification_report(csv_file_path, save_path='table/cnn_classification_report.png')

    csv_file_path = 'csv/svm_classification_report.csv'
    visualize_classification_report(csv_file_path, save_path='table/svm_classification_report.png')

    csv_file_path = 'csv/knn_classification_report.csv'
    visualize_classification_report(csv_file_path, save_path='table/knn_classification_report.png')