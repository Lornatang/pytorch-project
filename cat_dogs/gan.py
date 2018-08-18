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

if not os.path.exists('../data/catdog/external'):
    os.mkdir('../data/catdog/external')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 1
num_epoch = 100
z_dimension = 100  # noise dimension

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder('../data/catdog/val',
                                     transform=transform)
data_loader = DataLoader(train_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 0),  # batch, 32, 28, 28
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, 2),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0),  # batch, 64, 14, 14
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, 2)  # batch, 64, 7, 7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0),  # batch, 64, 14, 14
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, 2)  # batch, 64, 7, 7
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 0),  # batch, 64, 14, 14
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, 2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(7680, 1024),
            nn.LeakyReLU(True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        """
        x: batch, width, height, channel=1
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 50, 3, 1, 0),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, 1, 0),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 10, 3, 1, 0),  # batch, 1, 28, 28
            nn.BatchNorm2d(10),
            nn.ReLU(True)
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(10, 1, 3, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 3, 128, 128)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        return x


D = Discriminator().to(device)  # discriminator model
G = Generator(z_dimension, 256 * 256).to(device)  # generator model

criterion = nn.BCEWithLogitsLoss().to(device)  # binary cross entropy

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)


def train():
    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(data_loader):
            num_img = 128
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


            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} D real: {:.6f}, D fake: {:.6f}'.format(epoch, num_epoch, d_loss.item(), g_loss.item(),real_scores.data.mean(), fake_scores.data.mean()))
            real_images = to_img(real_img.data)
            save_image(real_images, '../data/catdog/external/real_images.jpg')

            fake_images = to_img(fake_img.data)
            save_image(fake_images, '../data/catdog/external/fake_images-{}.jpg'.format(epoch + 1))


if __name__ == '__main__':
    train()
