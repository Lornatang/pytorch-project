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
    def __init__(self):
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
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=512 * 6 * 6, out_features=512, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.75),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=2, bias=True),
        )

    def forward(self, x):
        out = self.features(x)

        dense = out.view(out.size(0), -1)

        out = self.classifier(dense)

        return out


base.train()