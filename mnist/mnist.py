"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: test.py
# time: 2018/8/6 23:38
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
parser.add_argument('--path', type=str, default='../data/mnist/',
                    help="""image path. Default='../data/mnist/'.""")
parser.add_argument('--epochs', type=int, default=20,
                    help="""num epochs. Default=10""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""0 ~ 9,. Default=10""")
parser.add_argument('--batch_size', type=int, default=100,
                    help="""batch size. Default=128""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--model_path', type=str, default='../../models/pytorch/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='mnist.pth',
                    help="""Model name""")
parser.add_argument('--display_epoch', type=int, default=1)
args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load data
train_data = datasets.MNIST(root=args.path,
                            train=True,
                            transform=train_transform,
                            download=True)

test_data = datasets.MNIST(root=args.path,
                           train=False,
                           transform=test_transform,
                           download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=args.batch_size,
                                          shuffle=False)


class CNN(nn.Module):
    
    def __init__(self, category=args.num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.Dropout(0.75),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.75),
            nn.ReLU(True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, category)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


# Load model
model = CNN().to(device)

print(CNN())
# cast
cast = nn.CrossEntropyLoss()
# Optimization
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def main():
    model.train()
    for epoch in range(1, args.epochs + 1):
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

        if epoch % args.display_epoch == 0 or epoch == 1:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

            # Test the model
            model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            correct = 0.
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Test Accuracy: {100 * correct/ total:.4f}")

    # Save the model checkpoint
    torch.save(model, args.model_path + args.model_name)


if __name__ == '__main__':
    main()
