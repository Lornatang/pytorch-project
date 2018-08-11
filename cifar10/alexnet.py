"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: alexnet.py
# time: 2018/8/9 02:02
# license: MIT
"""

import argparse
import os

import time
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/cifar10',
                    help="""image path. Default='../data/cifar10'.""")
parser.add_argument('--epochs', type=int, default=100,
                    help="""num epochs. Default=100""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""0 ~ 9,. Default=10""")
parser.add_argument('--batch_size', type=int, default=128,
                    help="""batch size. Default=128""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--model_path', type=str, default='../../model/pytorch/cifar10/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='alexnet.pth',
                    help="""Model name""")
parser.add_argument('--display_epoch', type=int, default=1)
args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=2),  # 先四周填充2，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B每层的归一化用到的均值和方差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Fashion mnist dataset
train_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                             train=True,
                                             transform=train_transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                            train=False,
                                            transform=test_transform,
                                            download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)


class AlexNet(nn.Module):
    def __init__(self, category=args.num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifical = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, category)
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifical(x)

        return x


# Load model
model = AlexNet().to(device)
print(AlexNet())
# cast
cast = nn.CrossEntropyLoss()
# Optimization
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def main():
    model.train()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.display_epoch == 0 or epoch == 1:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start):.1f}sec!")

            # Test the model
            model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Test Accuracy: {(correct / 100):.2f}%")

    # Save the model checkpoint
    torch.save(model, args.model_path + args.model_name)


if __name__ == '__main__':
    main()
"""
Acc: 71.95%
"""
