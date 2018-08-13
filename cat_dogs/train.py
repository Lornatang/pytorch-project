"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: train.py
# time: 2018/8/13 09:23
# license: MIT
"""

import argparse
import os

import sys
import torch
import torchvision
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='../data/catdog/',
                    help="""image dir path default: '../data/catdog/'.""")
parser.add_argument('--epochs', type=int, default=50,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size', type=int, default=5,
                    help="""Batch_size default:5.""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--model_path', type=str, default='../../model/pytorch/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='catdog.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=2)

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(128),  # 将图像转化为800 * 800
    transforms.RandomHorizontalFlip(p=0.75),  # 有0.75的几率随机旋转
    transforms.RandomCrop(114),  # 从图像中裁剪一个24 * 24的
    transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    # transforms.Grayscale(),  # 转化为灰度图
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])

# Load data
train_datasets = torchvision.datasets.ImageFolder(root=args.path + 'train/',
                                                  transform=transform)
test_datasets = torchvision.datasets.ImageFolder(root=args.path + 'test/',
                                                 transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=args.batch_size,
                                          shuffle=True)


def test():
    print(f"test numbers: {len(test_datasets)}.")
    model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model.eval()

    correct_prediction = 0.
    total = 0
    for images, labels in test_loader:
        # to GPU
        images = images.to(device)
        labels = labels.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)
        # val_loader total
        total += labels.size(0)
        # add correct
        correct_prediction += (predicted == labels).sum().item()

        print(labels)

    print(f"Acc: {(correct_prediction / total):4f}")


def val():
    val_datasetss = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                     transform=transform)
    val_loaders = torch.utils.data.DataLoader(dataset=val_datasetss)
    model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model.eval()

    for images, _ in val_loaders:
        # to GPU
        images = images.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)

        if predicted[0] == 0:
            print('is cat!')
        else:
            print('is dog!')


if __name__ == '__main__':
    if sys.argv[1] == '--test':
        test()
    elif sys.argv[1] == '--val':
        val()
