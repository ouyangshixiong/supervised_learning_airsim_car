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
#    基础模型(baseline)
#    By AlexOuyang
############################################################################################

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class CarBaselineModel(nn.Layer):

    def __init__(self):
        super(CarBaselineModel, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding="SAME")
        self.conv2 = nn.Conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding="SAME")
        self.conv3 = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding="SAME")
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.max_pool1 = nn.MaxPool2D(2)
        self.max_pool2 = nn.MaxPool2D(2)
        self.max_pool3 = nn.MaxPool2D(2)
        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(6944,1280)
        self.linear_2 = nn.Linear(1281,64)
        self.linear_3 = nn.Linear(64,1)


    def forward(self, img, states):
        img = paddle.divide(img, paddle.to_tensor(255).astype('float32'))
        img_stack = self.max_pool1(F.relu(self.conv1(img)))
        img_stack = self.max_pool2(F.relu(self.conv2(img_stack)))
        img_stack = self.max_pool3(F.relu(self.conv3(img_stack)))
        img_stack = self.flatten(img_stack)
        img_stack = self.dropout1(img_stack)

        # merged = paddle.concat(x=[img_stack, states], axis=1)

        merged = self.linear_1(img_stack)
        merged = self.dropout2(merged)
        merged = paddle.concat(x=[merged, states], axis=1)
        merged = self.linear_2(merged)
        merged = self.dropout3(merged)
        merged = self.linear_3(merged)
        return merged

model = CarBaselineModel()

paddle.summary(model, [(1, 3, 59, 255), (1, 1)], 
                                        dtypes=['float32', 'float32'])