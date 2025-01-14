"""
Code to train a DCGAN on the datasets: 'mnist', 'turbulence', 'turbulence_flow' or 'lorenz_attractor'
Implementation by Claudia Drygala, University of Wuppertal

Implementation follows description of the following repos:
1) Basic implementation to train a Vanilla-GAN: https://github.com/diegoalejogm/gans accompanied by a blog post
https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
2) Implementation of the DCGAN architecture: https://github.com/eriklindernoren/PyTorch-GAN

According to the original DCGAN paper the fundamental steps to train a GAN are: (https://arxiv.org/pdf/1511.06434.pdf)
1) Sample a noise set and a real-data set, each with size m.
2) Train the Discriminator on this data.
3) Sample a different noise subset with size m.
4) Train the Generator on this data.
5) Repeat from Step 1.
"""

import torch
from torch import nn, optim, Tensor
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
from utils import Logger
import time
import os
from options.train_options import TrainOptions

# We always train on a gpu!
cuda = True

# Parse the training options
opt = TrainOptions().parse()

# Define the gpu the code should be performed on
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

# Set the image size, since we need it multiple times in the code
img_size = opt.resize_img

# Load the dataset
# Datasets should have the following structure: data_folder->data_subfolder->images
# Datapath is the path to first level of the folders, so: \path_to\data_folder
def turbulence_data(resize_size):
    path_turbulence = opt.dataset_dir
    compose = transforms.Compose([transforms.Grayscale(), transforms.Resize(size=(resize_size, resize_size)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                                  ])
    return datasets.ImageFolder(root=path_turbulence, transform=compose)

# Create data loader, such that it is possible to iterate over data
data = turbulence_data(resize_size=img_size)
data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True)
# data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=False)

# Number of batches
num_batches = len(data_loader)

# Function to initalize weights if no pretraining will be applied
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Convert output of generator to image
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, img_size, img_size)

# Generate random noise vector sampled from a normal distribution
def noise(num_test_samples):
    n = Variable(torch.randn(num_test_samples, opt.latent_dim))
    return n.cuda()

# Definition of the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Definition of the generator
# Generators input is a latent variable vector
# Output: A n_out valued vector which corresponds to the flattened vector of the given images
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Initialize D and G
discriminator = Discriminator()
generator = Generator()

discriminator.cuda()
generator.cuda()

# Initialize weights
if opt.pretrained or opt.continue_train:
    state_g = torch.load(os.path.join(opt.checkpoint_dir, opt.model_G))
    generator.load_state_dict(state_g)
    state_d = torch.load(os.path.join(opt.checkpoint_dir, opt.model_D))
    discriminator.load_state_dict(state_d)
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizer
d_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
g_optimizer = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

# Loss function
loss = torch.nn.BCELoss()
loss.cuda()

# Create noise vector for validation during training
test_noise = noise(opt.num_test_samples)

# Create logger instance for training
logger = Logger(model_name='DCGAN', data_name=opt.experiment_name)

# Training routine for the GAN
Tensor = torch.cuda.FloatTensor
num_epochs = opt.num_epochs
for epoch in range(num_epochs):
    if opt.continue_train:
        epoch_count = opt.epoch_start + epoch
    else:
        epoch_count = epoch
    print('Epoch:', epoch_count)
    # Compute the time of one epoch
    start = time.time()
    for n_batch, (real_batch,_) in enumerate(data_loader):
        # Adversarial ground truths
        valid = Variable(Tensor(real_batch.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(real_batch.shape[0], 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_data = Variable(real_batch.type(Tensor))

        # Train G
        g_optimizer.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (real_batch.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_data = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_error = loss(discriminator(gen_data), valid)
        # print('Loss of G:', g_error)

        g_error.backward()
        g_optimizer.step()

        # Train D
        d_optimizer.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = loss(discriminator(real_data), valid)
        fake_loss = loss(discriminator(gen_data.detach()), fake)
        d_error = (real_loss + fake_loss) / 2
        # print('Loss of D:', d_error)

        d_error.backward()
        d_optimizer.step()

        # Log batch error
        logger.log(d_error, g_error, epoch_count, n_batch, num_batches)

        # Display Progress every few batches
        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(test_images, opt.num_test_samples, epoch_count, n_batch, num_batches)
            # Display status Logs
            if opt.continue_train:
                logger.display_status(epoch_count, num_epochs+opt.epoch_start, n_batch, num_batches, d_error, g_error)
            else:
                logger.display_status(epoch_count, num_epochs, n_batch, num_batches, d_error, g_error)
        # Save checkpoints
        if epoch % 10 == 0:
            logger.save_models(generator, discriminator, epoch_count)
        if epoch == (num_epochs - 1):
            logger.save_models(generator, discriminator, epoch_count)
    end = time.time()
    print('Computational time for epoch', str(epoch_count), ':', end - start)
