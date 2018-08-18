"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: gan.py
# time: 2018/7/30 07:27
# license: MIT
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

if not os.path.exists('../data/mnist/dc_img'):
    os.mkdir('../data/mnist/dc_img')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100  # noise dimension

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.MNIST('../data/mnist',
                               transform=transform,
                               download=True)
data_loader = DataLoader(train_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: batch, width, height, channel=1
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


D = Discriminator().to(device)  # discriminator model
G = Generator(z_dimension, 3136).to(device)  # generator model

criterion = nn.BCELoss().to(device)  # binary cross entropy

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)


def train():
    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(data_loader):
            num_img = img.size(0)
            # =================train discriminator
            real_img = img.to(device)
            real_label = torch.ones(num_img).to(device)
            fake_label = torch.zeros(num_img).to(device)

            # compute loss of real_img
            real_out = D(real_img)
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out  # closer to 1 means better

            # compute loss of fake_img
            z = torch.randn(num_img, z_dimension).to(device)
            fake_img = G(z)
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  # closer to 0 means better

            # bp and optimize
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ===============train generator
            # compute loss of fake_img
            z = torch.randn(num_img, z_dimension).to(device)
            fake_img = G(z)
            output = D(fake_img)
            g_loss = criterion(output, real_label)

            # bp and optimize
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                      'D real: {:.6f}, D fake: {:.6f}'
                      .format(epoch, num_epoch, d_loss.data[0], g_loss.data[0],
                              real_scores.data.mean(), fake_scores.data.mean()))
            if epoch == 0:
                real_images = to_img(real_img.data)
                save_image(real_images, '../data/mnist/dc_img/real_images.jpg')

            fake_images = to_img(fake_img.data)
            save_image(fake_images, '../data/mnist/dc_img/fake_images-{}.jpg'.format(epoch + 1))



if __name__ == '__main__':
    train()