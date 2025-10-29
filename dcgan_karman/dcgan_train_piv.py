"""
DCGAN training on PIV dataset (with u and v channels)
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import os
import time

from utils_ilmenau import Logger
from options.train_options import TrainOptions

# ------------------------------
#   Device setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# ------------------------------
#   Parse training options
# ------------------------------
opt = TrainOptions().parse()
img_size = opt.resize_img  # e.g., 256

# ------------------------------
#   Load .pt data
# ------------------------------
data_dict = torch.load(opt.dataset_dir)  # <-- path to your .pt file
outer_key = list(data_dict.keys())[opt.outer_key]
print("Using dataset:", outer_key)

u = data_dict[outer_key]["u"]  # (H, W, T)
v = data_dict[outer_key]["v"]
print("Loaded u:", u.shape, "v:", v.shape)

# ------------------------------
#   Dataset wrapper with top/right padding or downsizing of images
# ------------------------------
class TurbulenceDataset(Dataset):
    def __init__(self, u, v, target_size=None):
        super().__init__()
        self.u = u
        self.v = v
        self.length = u.shape[-1]  # number of time steps
        self.H_orig, self.W_orig = u.shape[0], u.shape[1]

        # Compute min/max for normalization to [0,1]
        self.u_min, self.u_max = float(u.min()), float(u.max())
        self.v_min, self.v_max = float(v.min()), float(v.max())

        self.target_size = target_size
        self.final_normalize = transforms.Normalize((0.5,0.5), (0.5,0.5))  # maps [0,1] -> [-1,1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # extract single time frame
        u_img = self.u[..., idx].float()
        v_img = self.v[..., idx].float()

        # normalize to [0,1]
        u_img = (u_img - self.u_min) / (self.u_max - self.u_min)
        v_img = (v_img - self.v_min) / (self.v_max - self.v_min)

        # stack channels
        img = torch.stack([u_img, v_img], dim=0)  # (2,H,W)

        if self.target_size:
            if self.target_size > self.H_orig or self.target_size > self.W_orig:
                # ---- pad up to larger square ----
                pad_h = self.target_size - self.H_orig
                pad_w = self.target_size - self.W_orig
                img = F.pad(img, (0, pad_w, 0, pad_h), "constant", 0.0)  # (C,H_target,W_target)
            elif self.target_size < self.H_orig or self.target_size < self.W_orig:
                # ---- resize down to smaller square ----
                img = TF.resize(img, (self.target_size, self.target_size), antialias=True)

        # normalize to [-1,1] for GAN
        img = self.final_normalize(img)
        return img

# ------------------------------
#   Denormalization helper
# ------------------------------
def denormalize(img_tensor, u_min, u_max, v_min, v_max, H_orig=None, W_orig=None):
    """
    img_tensor: GAN output in [-1,1], shape (N,2,H_target,W_target)
    Returns tensor scaled back to original u,v ranges, optionally resized back to original H/W
    """
    img_tensor = (img_tensor + 1.0) / 2.0  # [-1,1] -> [0,1]
    u = img_tensor[:,0,:,:] * (u_max - u_min) + u_min
    v = img_tensor[:,1,:,:] * (v_max - v_min) + v_min

    # resize back to original resolution if given
    if H_orig is not None and W_orig is not None:
        u = TF.resize(u.unsqueeze(1), (H_orig, W_orig), antialias=True).squeeze(1)
        v = TF.resize(v.unsqueeze(1), (H_orig, W_orig), antialias=True).squeeze(1)

    return torch.stack([u,v], dim=1)

# ------------------------------
#   Create dataset and dataloader
# ------------------------------
dataset = TurbulenceDataset(u, v, target_size=img_size)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

num_batches = len(data_loader)
print(f"Dataset length: {len(dataset)} | Num batches: {num_batches}")

# ------------------------------
#   Weight init
# ------------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# ------------------------------
#   Noise generator
# ------------------------------
def noise(num_samples):
    return Variable(torch.randn(num_samples, opt.latent_dim).to(device))

# ------------------------------
#   Discriminator
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_size=opt.resize_img, channels=opt.channels):
        super().__init__()
        def block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))  # better than BatchNorm
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(channels, 32, normalize=False),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256),
        )
        # compute feature map size dynamically
        ds_size = img_size // 2**4
        self.adv_layer = nn.Linear(256 * ds_size**2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)  # flatten
        return self.adv_layer(out)  # no sigmoid (using BCEWithLogitsLoss)


# ------------------------------
#   Generator
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=opt.latent_dim, img_size=opt.resize_img, channels=opt.channels):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size**2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

# ------------------------------
#   Initialize models
# ------------------------------
discriminator = Discriminator().to(device)
generator = Generator().to(device)

if opt.pretrained or opt.continue_train:
    state_g = torch.load(os.path.join(opt.checkpoint_dir, opt.model_G))
    generator.load_state_dict(state_g)
    state_d = torch.load(os.path.join(opt.checkpoint_dir, opt.model_D))
    discriminator.load_state_dict(state_d)
else:
    print("Random init of weights...")
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# ------------------------------
#   Optimizers & Loss
# ------------------------------
d_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
g_optimizer = optim.Adam(generator.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
loss_fn = nn.BCEWithLogitsLoss().to(device)

# ------------------------------
#   Logger
# ------------------------------
test_noise = noise(opt.num_test_samples)
logger = Logger(model_name="DCGAN", data_name=opt.experiment_name)

# ------------------------------
#   Training Loop
# ------------------------------
Tensor = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor
num_epochs = opt.num_epochs

for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs}")

    for n_batch, real_imgs in enumerate(data_loader):
        real_imgs = real_imgs.to(device)

        valid = Variable(Tensor(real_imgs.size(0), 1).uniform_(opt.lower_real, 1.0), requires_grad=False)
        fake = Variable(Tensor(real_imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # --- Train Generator ---
        g_optimizer.zero_grad()
        z = noise(real_imgs.size(0))
        gen_imgs = generator(z)
        g_loss = loss_fn(discriminator(gen_imgs), valid)
        g_loss.backward()
        g_optimizer.step()

        # --- Train Discriminator ---
        d_optimizer.zero_grad()
        real_loss = loss_fn(discriminator(real_imgs), valid)
        fake_loss = loss_fn(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # --- Logging ---
        logger.log(d_loss, g_loss, epoch, n_batch, num_batches)

        if n_batch % 20 == 0:
            test_imgs = generator(test_noise).data
            test_imgs_orig = denormalize(test_imgs, dataset.u_min, dataset.u_max, dataset.v_min, dataset.v_max,
                                         H_orig=dataset.H_orig, W_orig=dataset.W_orig)
            logger.log_images(test_imgs_orig, opt.num_test_samples, epoch, n_batch, num_batches)
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_loss, g_loss)

        if epoch % 10 == 0:
            logger.save_models(generator, discriminator, epoch)
