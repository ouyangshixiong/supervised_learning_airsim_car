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
#    预处理原始数据
#    写入到train.txt 和 test.txt
#    By AlexOuyang
############################################################################################

import os
import cv2
import numpy as np
import pandas as pd
import random

def load(relative_path):
    txt_file = os.path.join(absolute_path, relative_path, TXT_FILE)
    datafile = pd.read_csv(txt_file, sep='\t')
    for i in range(1, datafile.shape[0] - 1, 1):
        states = list(datafile.iloc[i-1][['Steering', 'Throttle', 'Brake', 'Speed (kmph)']])
        states = ",".join(str(i) for i in states)
        # 取前后两行的steering数据跟自身做平均
        label = list((datafile.iloc[i][['Steering']] + datafile.iloc[i-1][['Steering']] + datafile.iloc[i+1][['Steering']]) / 3.0)
        if label[0] == 0:
            r = random.randint(1,100)
            # drop 大部分标签为0的数据，让数据保持平衡
            if r < 95:
                continue
        label = ",".join(str(i) for i in label)
        image_path = os.path.join(os.path.join(absolute_path, relative_path, 'images'), datafile.iloc[i]['ImageName']).replace('\\', '/')
        line = image_path + "," + states  + ',' + label + '\n'
        if i % 100 >= 90:
            test_data.append(line)
        else:
            train_data.append(line)


RAW_DATA_DIR = 'data_raw'
DATA_DIR = 'data'
DATA_FOLDERS = ['normal_1', 'normal_2', 'normal_3', 'normal_4', 'normal_5', 'normal_6', 'swerve_1', 'swerve_2', 'swerve_3']
TXT_FILE = "airsim_rec.txt"
absolute_path = os.getcwd()
train_data = []
test_data = []
for folder in DATA_FOLDERS:
    relative_path = os.path.join(RAW_DATA_DIR, folder)
    load(relative_path)
    print('file:' + relative_path + "\\" + TXT_FILE + " loaded!")

if os.path.exists(DATA_DIR) == False:
    os.mkdir(DATA_DIR)

with open(os.path.join(DATA_DIR, "airsim_car_train.txt"), mode='w', encoding='utf-8') as f: 
    f.writelines(train_data)
with open(os.path.join(DATA_DIR, "airsim_car_test.txt"), mode='w', encoding='utf-8') as f: 
    f.writelines(test_data)