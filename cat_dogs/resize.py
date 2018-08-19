"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: resize.py
# time: 2018/8/18 22:16
# license: MIT
"""

import os

import cv2

for file in os.listdir('../data/catdog/train/dogs/'):
    file = os.path.join('../data/catdog/train/dogs/', file)
    img = cv2.imread(file)
    img = cv2.resize(img, (128, 128))
    cv2.imwrite(file, img)