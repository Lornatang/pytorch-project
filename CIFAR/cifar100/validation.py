"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: validation.py
# time: 2018/8/18 10:31
# license: MIT
"""

import argparse
import os

import torch
import torchvision
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='../../data/CIFAR/cifar100/',
                    help="""image dir path default: '../../data/CIFAR/cifar100/'.""")
parser.add_argument('--batch_size', type=int, default=1,
                    help="""Batch_size default:1.""")
parser.add_argument('--model_path', type=str, default='../../../models/pytorch/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='cifar100.pth',
                    help="""Model name.""")

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(32),  # 将图像转化为32 * 32
    transforms.RandomCrop(24),  # 从图像中裁剪一个114 * 114的
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])
# Load data
train_datasets = torchvision.datasets.ImageFolder(root=args.path,
                                                  transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=args.batch_size,
                                           shuffle=True)

val_datasets = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                transform=transform)


val_loader = torch.utils.data.DataLoader(dataset=val_datasets,
                                         batch_size=args.batch_size,
                                         shuffle=True)
# train_datasets dict
item = train_datasets.class_to_idx


def main():
    # Load model
    if torch.cuda.is_available():
        model = torch.load(args.model_path + args.model_name).to(device)
    else:
        model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model.eval()
    for i, (images, _) in enumerate(val_loader):
        # to GPU
        images = images.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)
        di = {v: k for k, v in item.items()}

        pred = di[int(predicted[0])]

        file = str(val_datasets.imgs[i])[2:-5]

        print(f"{i+1}.({file}) is {pred}!")


if __name__ == '__main__':
    main()
