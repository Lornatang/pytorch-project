"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: gan.py
# time: 2018/8/10 07:27
# license: MIT
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--path_dir', type=str, default='../data/catdog/extera',
                    help="""input image path dir.Default: '../data/catdog/extera'.""")
parser.add_argument('--external_dir', type=str, default='../data/catdog/external_data/',
                    help="""input image path dir.Default: '../data/catdog/external_data/'.""")
parser.add_argument('--latent_size', type=int, default=128,
                    help="""Latent_size. Default: 128.""")
parser.add_argument('--hidden_size', type=int, default=256,
                    help="""Hidden size. Default: 256.""")
parser.add_argument('--batch_size', type=int, default=128,
                    help="""Batch size. Default: 128.""")
parser.add_argument('--image_size', type=int, default=128 * 128 * 3,
                    help="""Input image size. Default: 128 * 128 * 3.""")
parser.add_argument('--max_epochs', type=int, default=500,
                    help="""Max epoch. Default: 500.""")
parser.add_argument('--display_epoch', type=int, default=2,
                    help="""When epochs save image. Default: 2.""")
args = parser.parse_args()

# Create a directory if not exists
if not os.path.exists(args.path_dir):
    os.makedirs(args.path_dir)

# Image processing
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # 3 for RGB channels

# MNIST dataset
train_dataset = torchvision.datasets.ImageFolder(root=args.path_dir,
                                                 transform=transform)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.image_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )

    def forward(self, x):
        out = self.fc(x)

        return out


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.latent_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.image_size),
            nn.ReLU())

    def forward(self, x):
        out = self.fc(x)

        return out


# Load model
D = torch.load('D.pth', map_location='cpu').to(device)
G = torch.load('G.pth', map_location='cpu').to(device)

# Binary cross entropy loss and optimizer
cast = nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, weight_decay=1e-8)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, weight_decay=1e-8)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
total_step = len(data_loader)
for epoch in range(1, args.max_epochs + 1):
    for i, (images, _) in enumerate(data_loader):
        start = time.time()
        images = images.reshape(images.size(0), -1).to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = cast(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(images.size(0), args.latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = cast(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(images.size(0), args.latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = cast(outputs, real_labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 32 == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{args.max_epochs}], "
                  f"Step [{i+1}/{total_step}], "
                  f"D_loss: {d_loss.item():.4f}, "
                  f"G_loss: {g_loss.item():.4f}, "
                  f"D(x): {real_score.mean().item():.2f}, "
                  f"D(G(z)): {fake_score.mean().item():.2f},"
                  f"Time: {(end-start):.2f}sec")

    # Save real images
    if epoch == 1:
        images = images.reshape(images.size(0), 3, 128, 128)
        save_image(denorm(images), os.path.join(args.external_dir, 'real_images.jpg'))

    if epoch % args.display_epoch == 0:
        # Save sampled images
        fake_images = fake_images.reshape(fake_images.size(0), 3, 128, 128)
        save_image(denorm(fake_images),
                   os.path.join(args.external_dir,
                   f"cat.{int(epoch / args.display_epoch + 4000)}.jpg"))

# Save the model checkpoints
torch.save(G, 'G.pth')
torch.save(D, 'D.pth')
