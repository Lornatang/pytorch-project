"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: train.py
# time: 2018/8/13 09:23
# license: MIT
"""

import os

import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../data/ANIMALS/cat_dogs/'
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

MODEL_PATH = '../../../models/pytorch/ANIMALS/cat_dogs/'
MODEL_NAME = 'catdog.pth'


# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize(224),  # 将图像转化为224 * 224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])


# Load data
train_datasets = torchvision.datasets.ImageFolder(root=WORK_DIR + 'train/',
                                                  transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_datasets = torchvision.datasets.ImageFolder(root=WORK_DIR + 'val/',
                                                 transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


class AlexNet(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),

            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out


def main():
    print(f"Train numbers:{len(train_datasets)}")
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUM_CLASSES),
        )
    # cost
    cost = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)

    for epoch in range(1, NUM_EPOCHS + 1):
        # model.train()
        # start time
        start = time.time()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * 2:.1f}sec!")

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

                _, predicted = torch.argmax(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            print(f"Acc: {(correct_prediction / total):4f}")

        # Save the model checkpoint
        torch.save(model, MODEL_PATH + MODEL_NAME)
    print(f"Model save to {MODEL_PATH + MODEL_NAME}.")


if __name__ == '__main__':
    main()
