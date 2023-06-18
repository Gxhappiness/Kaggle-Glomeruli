# -*- coding = utf-8 -*-
# @Time : 2023/6/17 17:57
# @Author : Happiness
# @File : split.py
# @Software : PyCharm




####导入工具包

import os
import random



#获取全部数据文件名列表
PATH_IMAGE = 'D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Glomeruli-dataset/images'
all_file_list = os.listdir(PATH_IMAGE)
all_file_num = len(all_file_list)
random.shuffle(all_file_list) # 随机打乱全部数据文件名列表



###指定训练集和测试集比例
train_ratio = 0.8
test_ratio = 1 - train_ratio
train_file_list = all_file_list[:int(all_file_num*train_ratio)]
test_file_list = all_file_list[int(all_file_num*train_ratio):]
print('数据集图像总数', all_file_num)
print('训练集划分比例', train_ratio)
print('训练集图像个数', len(train_file_list))
print('测试集图像个数', len(test_file_list))




train_file_list[:5]
['SAS_21883_001_35.png',
 'VUHSK_1352_59.png',
 'SAS_21908_001_60.png',
 'SESCAM_9_0_25.png',
 'SAS_21896_001_26.png']
test_file_list[:5]
['VUHSK_1272_101.png',
 'SAS_21937_001_117.png',
 'VUHSK_1502_11.png',
 'SAS_21904_001_3.png',
 'VUHSK_1502_8.png']



###生成两个txt划分文件
with open('D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Glomeruli-dataset/splits/train.txt', 'w') as f:
    f.writelines(line.split('.')[0] + '\n' for line in train_file_list)
with open('D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Glomeruli-dataset/splits/val.txt', 'w') as f:
    f.writelines(line.split('.')[0] + '\n' for line in test_file_list)