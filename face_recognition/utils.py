"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: utils.py
# time: 2018/8/15 12:45
# license: MIT
"""

import argparse
import os

import cv2
import dlib

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/face/val/',
                    help="""Train path dir.""")
args = parser.parse_args()


def get_face_location():
    detector = dlib.get_frontal_face_detector()
    for path_dir in os.listdir(args.path):
        for files in os.listdir(args.path + path_dir + '/'):
            #  图片路径
            file = os.path.join(args.path, path_dir, files)
            # 读取图片
            img = cv2.imread(file)
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1, x2:y2]
                # 调整图片大小
                face = cv2.resize(face, (128, 128))

                cv2.imwrite(f"{file}", face)


get_face_location()


