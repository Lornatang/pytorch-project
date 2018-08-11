"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: vgg11.py
# time: 2018/8/2 07:02
# license: MIT
"""

import argparse
import os

import time
import torch
import torchvision
from torch import nn
from torchvision import transforms

# Configuration
device = ('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/fashion',
                    help="""image path. Default='../data/fashion'.""")
parser.add_argument('--epochs', type=int, default=20,
                    help="""num epochs. Default=200""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""0 ~ 9,. Default=10""")
parser.add_argument('--batch_size', type=int, default=100,
                    help="""batch size. Default=100""")
parser.add_argument('--lr', type=float, default=0.001,
                    help="""learing_rate. Default=0.001""")
parser.add_argument('--model_path', type=str, default='../../model/pytorch/mnist/fashion_mnist/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='vgg11.pth',
                    help="""Model name""")
parser.add_argument('--display_epoch', type=int, default=2)
args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# Define transforms.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Fashion mnist dataset
train_dataset = torchvision.datasets.FashionMNIST(root=args.path,
                                                  train=True,
                                                  transform=train_transform,
                                                  download=True)

test_dataset = torchvision.datasets.FashionMNIST(root=args.path,
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


class Vgg11(nn.Module):
    def __init__(self, category=args.num_classes):
        super(Vgg11, self).__init__()
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),

            # Conv 4
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),

            # Conv 5
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),

            # Conv 6
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),

            # Conv 7
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),

            # Conv 8
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            # Fc 1
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # Fc 2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # Out
            nn.Linear(4096, category)

        )

    def forward(self, x):
        out = self.features(x)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out


# Load model
model = Vgg11()
print(model)

# Correct
cast = nn.CrossEntropyLoss()

# Optmizeation
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def main():
    model.train()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
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
            acc = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                acc += (predicted == labels).sum().item()

            print(f"Test Accuracy: {(acc / 100):.2f}%")

        # Save the model checkpoint
        torch.save(model, args.model_path + args.model_name)


if __name__ == '__main__':
    main()
