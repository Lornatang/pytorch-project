"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: gan.py
# time: 2018/7/30 07:27
# license: MIT
"""

import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import cv2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parmeters
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='../../data',
                    help="""image path. Default '../data'.""")
parser.add_argument('--sample_dir', type=str, default='samples',
                    help="""Create a directory if not exists. Default 'samples""")
parser.add_argument('--latent_size', type=int, default=64,
                    help="""Latent size. Default 64.""")
parser.add_argument('--hidden_size', type=int, default=256,
                    help="""Hidden size. Default 256.""")
parser.add_argument('--image_size', type=int, default=784,
                    help="""Image size. Default 784.""")
parser.add_argument('--num_epochs', type=int, default=200,
                    help="""num epochs. Default 200.""")
parser.add_argument('--batch_size', type=int, default=300,
                    help="""batch size. Default 300.""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default 0.0001.""")
args = parser.parse_args()


# Create a dir if not exists
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)

# Img processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root=args.dir,
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=args.batch_size,
                                          shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(args.image_size, args.hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(args.hidden_size, args.hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(args.hidden_size, 1)
)

# Generator
G = nn.Sequential(
    nn.Linear(args.latent_size, args.hidden_size),
    nn.ReLU(),
    nn.Linear(args.hidden_size, args.hidden_size),
    nn.ReLU(),
    nn.Linear(args.hidden_size, args.image_size),
    nn.Tanh()
)

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
correct_prediction = nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, weight_decay=1e-8)
g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, weight_decay=1e-8)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
total_step = len(data_loader)
for epoch in range(args.num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(args.batch_size, -1).to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(args.batch_size, 1).to(device)
        fake_labels = torch.zeros(args.batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = correct_prediction(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(args.batch_size, args.latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = correct_prediction(outputs, fake_labels)
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
        z = torch.randn(args.batch_size, args.latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = correct_prediction(outputs, real_labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print(f"Epoch {epoch} d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, "
                  f"D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}")

    # Save real images
    if (epoch + 1) == 1:
        images = cv2.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(args.sample_dir, 'real_images.png'))

    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(args.sample_dir, 'fake_images-{}.png'.format(epoch + 1)))
