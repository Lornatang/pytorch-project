"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: train.py
# time: 2018/8/13 09:23
# license: MIT
"""

import argparse
import os
import time

import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='../data/catdog/',
                    help="""image dir path default: '../data/catdog/'.""")
parser.add_argument('--epochs', type=int, default=50,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size', type=int, default=64,
                    help="""Batch_size default:64.""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--num_classes', type=int, default=2,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='../../model/pytorch/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='catdog.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=5)

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

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=args.batch_size,
                                           shuffle=True)


class Net(nn.Module):
    def __init__(self, category=args.num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)  # Size / 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # Size / 2

        self.layer1 = nn.Sequential(
            # BasicBlock 1
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),

            # BasicBlock 2
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )  # Size / 2

        self.layer2 = nn.Sequential(
            # BasicBlock 1
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),

            # downsample
            nn.Conv2d(64, 128, 1, 2, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # BasicBlock 2
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128)
        )  # Size / 2

        self.layer3 = nn.Sequential(
            # BasicBlock 1
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),

            # downsample
            nn.Conv2d(128, 256, 1, 2, 0),
            nn.BatchNorm2d(256),

            # BasicBlock 2
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )  # Size / 2

        self.layer4 = nn.Sequential(
            # BasicBlock 1
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.atchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),

            # downsample
            nn.Conv2d(256, 512, 1, 2),
            nn.BatchNorm2d(512),

            # BasicBlock 2
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )  # Size / 2

        self.avgpool = nn.AvgPool2d(4, 1)  # Size = (Size-7+1)

        self.fc = nn.Linear(512, category)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        dense = out.reshape(out.size(0), -1)
        out = self.fc(dense)

        return out


def train():
    print(f"Train numbers:{len(train_datasets)}")

    # Load model
    # if torch.cuda.is_available():
    #     model = torch.load(args.model_path + args.model_name).to(device)
    # else:
    #     model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model = Net().to(device)
    print(model)
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        # start time
        start = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.display_epoch == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

    # Save the model checkpoint
    torch.save(model, args.model_path + args.model_name)
    print(f"Model save to {args.model_path + args.model_name}.")


if __name__ == '__main__':
    train()
