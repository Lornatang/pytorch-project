"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: lenet5.py
# time: 2018/8/9 20:17
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
parser.add_argument('--epochs', type=int, default=10,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size', type=int, default=128,
                    help="""Batch_size default:2.""")
parser.add_argument('--num_classes', type=int, default=2,
                    help="""0 ~ 1,. Default=2""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--model_path', type=str, default='../../model/pytorch/catdog/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='lenet5.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=1)
args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像转化为800 * 800
    transforms.RandomHorizontalFlip(p=0.75),  # 有0.75的几率随机旋转
    # transforms.RandomCrop(54),  # 从图像中裁剪一个24 * 24的
    transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    # transforms.Grayscale(),  # 转化为灰度图
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])


train_datasets = torchvision.datasets.ImageFolder(root=args.path + 'train/',
                                                  transform=transform,)
val_datasets = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=args.batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_datasets,
                                         batch_size=args.batch_size,
                                         shuffle=True)


class Net(torch.nn.Module):
    def __init__(self, category=args.num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dense = nn.Sequential(
            nn.Linear(256 * 10 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(128, category)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.reshape(conv4_out.size(0), -1)
        out = self.dense(res)
        return out


# Load model
model = torchvision.models.alexnet(pretrained=True).to(device)

model.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 2),
    )
print(model)
# cast
cast = nn.CrossEntropyLoss()
# Optimization
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def main():
    print(f"Trian numbers:{len(train_datasets)}")
    print(f"Val numbers:{len(val_datasets)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
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
            test()

    # Save the model checkpoint
    model_path = torch.save(model, args.model_path + args.model_name)
    print(f"Model save to {model_path}.")


def test():
    model.eval()
    correct = 0.
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Acc: {correct / total:f}")


if __name__ == '__main__':
    test()

