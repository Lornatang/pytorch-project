"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: validation.py
# time: 2018/8/13 10:12
# license: MIT
"""

# encoding=utf-8

import argparse
import numpy as np

import torch
import torch.nn
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../../model/pytorch/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='catdog.pth',
                    help="""Model name.""")
args = parser.parse_args()

img_to_tensor = transforms.ToTensor()


def make_model():
    resmodel = torch.load(args.model_path + args.model_name)
    resmodel.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return resmodel


# 分类
def inference(resmodel, imgpath):
    resmodel.eval()  # 必需，否则预测结果是错误的

    img = Image.open(imgpath)
    img = img.resize((128, 128))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 128, 128)
    tensor = tensor.cuda()  # 将数据发送到GPU，数据和模型在同一个设备上运行

    result = resmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 将结果传到CPU，并转换为numpy格式
    max_index = np.argmax(result_npy[0])

    return max_index


# 特征提取
def extract_feature(resmodel, imgpath):
    resmodel.fc = torch.nn.LeakyReLU(0.1)
    resmodel.eval()

    img = Image.open(imgpath)
    img = img.resize((128, 128))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 128, 128)
    tensor = tensor.cuda()

    result = resmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()

    return result_npy[0]


if __name__ == "__main__":
    model = make_model()
    imgpath = '../data/catdog/train/cats/cat.1.jpg'
    print(inference(model, imgpath))
    print(extract_feature(model, imgpath))
