#############################################################################################
#    MIT License
#
#    Copyright (c) AlexOuyang. All rights reserved.
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE
############################################################################################

############################################################################################
#    编写自定义DataSet、自定义DataLoader
#    支持参数train和test
#    By AlexOuyang
############################################################################################

import os
import cv2
import math
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import ColorJitter, RandomHorizontalFlip
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from part_1_explore_image import draw_image_with_label

TRAIN_DATA = "data/airsim_car_train.txt"
TEST_DATA = "data/airsim_car_test.txt"

ROI = [76, 135, 0, 255]

absolute_path = os.getcwd()
color_transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
flip_transform = RandomHorizontalFlip(0.5)

class AirSimCarDataSet(Dataset):
    def __init__(self, mode='train'):
        super(AirSimCarDataSet, self).__init__()
        self.train_data = []
        self.test_data = []
        with open(TRAIN_DATA,encoding='utf-8') as f:
            for line in f.readlines():
                 image_path, s1, s2, s3, s4, label = line.strip().split(',')
                 self.train_data.append([image_path, np.array(s1).astype('float32'), np.array(label).astype('float32') ])
        with open(TEST_DATA,encoding='utf-8') as f:
            for line in f.readlines():
                 image_path, s1, s2, s3, s4, label = line.strip().split(',')
                 self.test_data.append([image_path, np.array([s1]).astype('float32'), np.array(label).astype('float32') ])
        if mode == 'test':
            self.data = self.test_data
        else:
            self.data = self.train_data

    def __getitem__(self, index):
        image_path, states, label = self.data[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = color_transform(image)
        image2 = flip_transform(image)
        # 翻转了
        equal = image == image2
        if equal.all():
            states = np.negative(states)
            label = np.negative(label)
        image2 = image2.astype('float32')
        image2 = image2[ ROI[0]:ROI[1], ROI[2]:ROI[3]]
        image2 = image2.transpose((2, 0, 1))
        return image2, states, label

    def __len__(self):
        return len(self.data)




# dataset1 = AirSimCarDataSet("test")
# dataset1 = AirSimCarDataSet("train")

# 定义并初始化数据读取器
# train_loader = paddle.io.DataLoader(dataset1, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

# 调用 DataLoader 迭代读取数据
# for batch_id, data in enumerate(train_loader()):
#     images, states, label = data
#     print("batch_id: {}, camera data shape: {}, 状态：{}, 标签: {}".format(batch_id, images[63].shape, states[63].numpy(), label[63].numpy()))
#     draw_image_with_label(images[63],label[63])
#     break