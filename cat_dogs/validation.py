"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: validation.py
# time: 2018/8/14 09:43
# license: MIT
"""

import argparse
import os

import torch
import torchvision
from torch import nn
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='../data/catdog/',
                    help="""image dir path default: '../data/catdog/'.""")
parser.add_argument('--batch_size', type=int, default=1,
                    help="""Batch_size default:64.""")
parser.add_argument('--num_classes', type=int, default=2,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='../../model/pytorch/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='catdog.pth',
                    help="""Model name.""")

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(128),  # 将图像转化为128 * 128
    transforms.RandomCrop(114),  # 从图像中裁剪一个114 * 114的
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])
# Load data
train_datasets = torchvision.datasets.ImageFolder(root=args.path + 'train/',
                                                  transform=transform)
val_datasets = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                transform=transform)


val_loader = torch.utils.data.DataLoader(dataset=val_datasets,
                                         batch_size=args.batch_size,
                                         shuffle=True)
# train_datasets zip
item = train_datasets.class_to_idx


class Net(nn.Module):
    def __init__(self, category=args.num_classes):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=4608, out_features=512, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.75),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=category, bias=True),
        )

    def forward(self, x):
        out = self.features(x)

        dense = out.reshape(out.size(0), -1)

        out = self.classifier(dense)

        return out


def val():
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
    val()
