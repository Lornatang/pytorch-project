"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: gan.py
# time: 2018/8/10 07:27
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
parser.add_argument('--img_dir', type=str, default='../data/catdog/extera/',
                    help="""input image path dir.Default: '../data/catdog/extera/'.""")
parser.add_argument('--external_dir', type=str, default='../data/catdog/external_data/',
                    help="""input image path dir.Default: '../data/catdog/external_data/'.""")
parser.add_argument('--latent_size', type=int, default=64,
                    help="""Latent_size. Default: 64.""")
parser.add_argument('--hidden_size', type=int, default=256,
                    help="""Hidden size. Default: 256.""")
parser.add_argument('--batch_size', type=int, default=400,
                    help="""Batch size. Default: 400.""")
parser.add_argument('--image_size', type=int, default=28 * 28 * 3,
                    help="""Input image size. Default: 28 * 28 * 3.""")
parser.add_argument('--max_epochs', type=int, default=100,
                    help="""Max epoch. Default: 100.""")
parser.add_argument('--display_epoch', type=int, default=5,
                    help="""When epochs save image. Default: 5.""")
parser.add_argument('--model_dir', type=str, default='../../models/pytorch/GAN/mnist/',
                    help="""Model save path dir. Default: '../../models/pytorch/GAN/mnist/'.""")
args = parser.parse_args()

# Create a directory if not exists
if not os.path.exists(args.external_dir):
    os.makedirs(args.external_dir)

# Image processing
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # 3 for RGB channels

# train dataset
train_dataset = datasets.ImageFolder(root=args.img_dir,
                                     transform=transform)

# Data loader
data_loader = data.DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

# Discriminator
Discriminator = nn.Sequential(
    nn.Linear(args.image_size, args.hidden_size),
    nn.ReLU(True),
    nn.Linear(args.hidden_size, args.hidden_size),
    nn.ReLU(True),
    nn.Linear(args.hidden_size, 1)
)

# Generator
Generator = nn.Sequential(
    nn.Linear(args.latent_size, args.hidden_size),
    nn.ReLU(True),
    nn.Linear(args.hidden_size, args.hidden_size),
    nn.ReLU(True),
    nn.Linear(args.hidden_size, args.image_size),
    nn.ReLU(True)
)

# Device setting
D = torch.load(args.model_dir + 'Discriminator.pth', map_location='cpu').to(device)
G = torch.load(args.model_dir + 'Generator.pth', map_location='cpu').to(device)

# Binary cross entropy loss and optimizer
cast = nn.BCEWithLogitsLoss().to(device)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, weight_decay=1e-5)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, weight_decay=1e-5)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
total_step = len(data_loader)
for epoch in range(1, args.max_epochs+1):
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
        d_loss_real = cast(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(args.batch_size, args.latent_size).to(device)
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
        z = torch.randn(args.batch_size, args.latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = cast(outputs, real_labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch}/{args.max_epochs}], "
                  f"Step [{i+1}/{total_step}], "
                  f"D_loss: {d_loss.item():.4f}, "
                  f"G_loss: {g_loss.item():.4f}, "
                  f"D(x): {real_score.mean().item():.2f}, "
                  f"D(G(z)): {fake_score.mean().item():.2f}")

    # Save real images
    if epoch == 1:
        images = images.reshape(images.size(0), 3, 28, 28)
        save_image(denorm(images), os.path.join(args.exteranal_data, 'origin.jpg'))

    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 3, 28, 28)
    save_image(denorm(fake_images), os.path.join(
        args.exteranal_data, f"cat.{4000 + epoch}.jpg"))

# Save the model checkpoints
# torch.save(G, 'G.ckpt')
# torch.save(D, 'D.ckpt')
