"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: lenet5.py
# time: 2018/8/9 01:05
# license: MIT
"""

import argparse
import os

import time
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/cifar10',
                    help="""image path. Default='../data/cifar10/'.""")
parser.add_argument('--epochs', type=int, default=10,
                    help="""num epochs. Default=10""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""0 ~ 9,. Default=10""")
parser.add_argument('--batch_size', type=int, default=100,
                    help="""batch size. Default=100""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--model_path', type=str, default='../../model/pytorch/cifar10/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='lenet5.pth',
                    help="""Model name""")
parser.add_argument('--display_epoch', type=int, default=1)
args = parser.parse_args()


# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=2),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B每层的归一化用到的均值和方差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_dataset = datasets.CIFAR10(root='../data/cifar10',
                                 train=True,
                                 transform=train_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='../data/cifar10',
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


class LeNet(nn.Module):
    def __init__(self, category=args.num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 2),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 120),
            nn.Linear(120, 84),
            nn.Linear(84, category)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out


# Laod model
model = LeNet().to(device)
print(model)
# Cast
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
Acc: 56.79%
"""