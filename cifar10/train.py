"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: train.py
# time: 2018/8/14 09:43
# license: MIT
"""

import base
from torch import nn


class Net(nn.Module):
    def __init__(self, category=args.num_classes):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=32*6*6, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.75),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(True),

            nn.Linear(in_features=128, out_features=category, bias=True),
        )

    def forward(self, x):
        out = self.features(x)

        dense = out.view(out.size(0), -1)

        out = self.classifier(dense)

        return out


if __name__ == '__main__':
    base.train()
