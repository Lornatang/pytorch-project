"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: gan.py
# time: 2018/8/15 07:27
# license: MIT
"""

import argparse
import os

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='../data/mnist/',
                    help="""input image path dir.Default: '../data/mnist/'.""")
parser.add_argument('--external_dir', type=str, default='../data/mnist/external_data/',
                    help="""input image path dir.Default: '../data/mnist/external_data/'.""")
parser.add_argument('--noise', type=int, default=100,
                    help="""Data noise. Default: 100.""")
parser.add_argument('--hidden_size', type=int, default=64,
                    help="""Hidden size. Default: 64.""")
parser.add_argument('--batch_size', type=int, default=64,
                    help="""Batch size. Default: 64.""")
parser.add_argument('--lr', type=float, default=2e-4,
                    help="""Train optimizer learning rate. Default: 2e-4.""")
parser.add_argument('--img_size', type=int, default=96,
                    help="""Input image size. Default: 96.""")
parser.add_argument('--max_epochs', type=int, default=50,
                    help="""Max epoch of train of. Default: 50.""")
parser.add_argument('--display_epoch', type=int, default=2,
                    help="""When epochs save image. Default: 2.""")
parser.add_argument('--model_dir', type=str, default='../../models/pytorch/GAN/mnist/',
                    help="""Model save path dir. Default: '../../models/pytorch/GAN/mnist/'.""")
args = parser.parse_args()

# Create a directory if not exists
if not os.path.exists(args.external_dir):
    os.makedirs(args.external_dir)

# Image processing
transform = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # 3 for RGB channels

# train dataset
train_dataset = datasets.MNIST(root=args.img_dir,
                               transform=transform,
                               download=True,
                               train=True)

# Data loader
data_loader = data.DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(args.hidden_size * 8)x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(args.noise, args.hidden_size * 8, 4, 1, 0),
            nn.BatchNorm2d(args.hidden_size * 8),
            nn.ReLU(True)
        )
        # layer2输出尺寸(args.hidden_size * 4)x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(args.hidden_size * 8, args.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(args.hidden_size * 4),
            nn.ReLU(True)
        )
        # layer3输出尺寸(args.hidden_size * 2)x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(args.hidden_size * 4, args.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(args.hidden_size * 2),
            nn.ReLU(True)
        )
        # layer4输出尺寸(args.hidden_size)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(args.hidden_size * 2, args.hidden_size, 4, 2, 1),
            nn.BatchNorm2d(args.hidden_size),
            nn.ReLU(True)
        )
        # layer5输出尺寸 1x96x96  BGR需要输入3x96x96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(args.hidden_size, 1, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    # 定义Generator的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# 定义鉴别器网络D
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # layer1 输入 1 x 96 x 96, 输出 (args.hidden_size) x 32 x 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, args.hidden_size, 5, 3, 1),
            nn.BatchNorm2d(args.hidden_size),
            nn.LeakyReLU(0.2, True)
        )
        # layer2 输出 (args.hidden_size * 2) x 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(args.hidden_size, args.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(args.hidden_size * 2),
            nn.LeakyReLU(0.2, True)
        )
        # layer3 输出 (args.hidden_size * 4) x 8 x 8
        self.layer3 = nn.Sequential(
            nn.Conv2d(args.hidden_size * 2, args.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(args.hidden_size * 4),
            nn.LeakyReLU(0.2, True)
        )
        # layer4 输出 (args.hidden_size * 8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(args.hidden_size * 4, args.hidden_size * 8, 4, 2, 1),
            nn.BatchNorm2d(args.hidden_size * 8),
            nn.LeakyReLU(0.2, True)
        )
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(args.hidden_size * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    # 定义NetD的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))


def train():
    label = torch.FloatTensor(args.batch_size)
    real_label = 1
    fake_label = 0
    for epoch in range(1, args.max_epochs + 1):
        for i, (img, _) in enumerate(data_loader):
            # 固定生成器G，训练鉴别器D
            optimizerD.zero_grad()
            # 让D尽可能的把真图片判别为1
            img = img.to(device)
            output = netD(img)

            label.data.fill_(real_label)
            label = label.to(device)
            errD_real = criterion(output, label)
            errD_real.backward()
            # 让D尽可能把假图片判别为0
            label.data.fill_(fake_label)
            noise = torch.randn(args.batch_size, args.noise, 1, 1)
            noise = noise.to(device)
            # 生成假图
            fake = netG(noise)
            output = netD(fake.detach())  # 避免梯度传到G，因为G不用更新
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_fake + errD_real
            optimizerD.step()

            # 固定鉴别器D，训练生成器G
            optimizerG.zero_grad()
            # 让D尽可能把G生成的假图判别为1
            label.data.fill_(real_label)
            label = label.to(device)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            print(f"Epoch: [{epoch}/{args.max_epochs}], "
                  f"Step: [{i}/{len(data_loader)}], "
                  f"Loss_D: {errD.item():.3f}, "
                  f"Loss_G {errG.item():.3f}.")

            save_image(fake.data,
                       f"{args.external_dir}/{epoch}.jpg",
                       normalize=True)

    # Save the model checkpoints
    torch.save(Generator, args.model_dir + 'Generator.pth')
    torch.save(Discriminator, args.model_dir + 'Discriminator.pth')


if __name__ == '__main__':
    train()
