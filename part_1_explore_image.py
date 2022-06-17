#############################################################################################
#    MIT License
#
#    Copyright (c) Microsoft Corporation. All rights reserved.
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
#    MIT 协议是一项常见的开源项目许可协议，被授权人有权利使用、复制、修改、合并、出版发行、
#    散布、再授权及贩售软件及软件的副本，但必须包含版权声明和许可声明。
#    By AlexOuyang
############################################################################################

import os
import sys
import cv2
import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

RAW_DATA_DIR = 'data_raw/'

absolute_path = os.getcwd()
sample_image_path = os.path.join(RAW_DATA_DIR, 'normal_1/images/img_0.png')
sample_image = Image.open(sample_image_path)

def showOriginal():
    plt.title('Sample Image')
    plt.imshow(sample_image)
    plt.show()

def showClipped():
    sample_image_roi = sample_image.copy()
    fillcolor=(255,0,0)
    draw = ImageDraw.Draw(sample_image_roi)
    points = [(1,76), (1,135), (255,135), (255,76)]
    #points = [(1,76), (1,135), (255,135)]
    for i in range(0, len(points), 1):
        draw.line([points[i], points[(i+1)%len(points)]], fill=fillcolor, width=3)
    del draw

    plt.title('Image with sample ROI')
    plt.imshow(sample_image_roi)
    plt.show()

def draw_image_with_label(img, label, prediction=None):
    np_label = label.numpy() * 0.1
    theta = np_label * 0.69 #Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 50
    line_thickness = 3
    label_line_color = (255, 0, 0)
    prediction_line_color = (0, 0, 255)
    img = img.numpy().transpose((1, 2, 0))
    pil_image = Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))

    print('Actual Steering Angle = {0}'.format(np_label))
    draw_image = pil_image.copy()
    image_draw = ImageDraw.Draw(draw_image)
    first_point = (int(img.shape[1]/2),img.shape[0])
    second_point = (int((img.shape[1]/2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
    image_draw.line([first_point, second_point], fill=label_line_color, width=line_thickness)

    if (prediction is not None):
        np_prediction = prediction.numpy() * 0.1
        print('Predicted Steering Angle = {0}'.format(np_prediction))
        print('L1 Error: {0}'.format(abs(np_prediction-np_label)))
        theta = np_prediction * 0.69
        second_point = (int((img.shape[1]/2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)
    
    del image_draw
    plt.imshow(draw_image)
    plt.show()
    cv2.waitKey()

def main(argv):
    if len(argv) < 2:
        showOriginal()
    else:
        showClipped()

if __name__=="__main__":
    main(sys.argv)