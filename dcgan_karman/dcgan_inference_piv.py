import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from options.inference_options import InferenceOptions

# ------------------------------
#  Device and options setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Inference on:", device)
opt = InferenceOptions().parse()

# ------------------------------
#   Generator (copied from training)
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size**2))
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
#   Denormalization helper
# ------------------------------
def denormalize(img_tensor, u_min, u_max, v_min, v_max, H_orig, W_orig):
    """
    img_tensor: GAN output in [-1,1], shape (N,2,H_gen,W_gen)
    Returns tensor scaled back to original u,v ranges.
    """
    img_tensor = (img_tensor + 1.0) / 2.0
    u = img_tensor[:, 0, :, :] * (u_max - u_min) + u_min
    v = img_tensor[:, 1, :, :] * (v_max - v_min) + v_min

    H_gen, W_gen = u.shape[-2], u.shape[-1]

    if H_gen >= H_orig and W_gen >= W_orig:
        # Training used padding → crop
        u = u[:, :H_orig, :W_orig]
        v = v[:, :H_orig, :W_orig]
    else:
        # Training used resizing → resize back
        u = TF.resize(u.unsqueeze(1), (H_orig, W_orig), antialias=True).squeeze(1)
        v = TF.resize(v.unsqueeze(1), (H_orig, W_orig), antialias=True).squeeze(1)

    return torch.stack([u, v], dim=1)

# ------------------------------
#   16-bit PNG saving helper
# ------------------------------
def save_uv_16bit_png(channel_array, out_dir, channel_name, min_val, max_val):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(channel_array.shape[0]):
        img = channel_array[i].astype(np.float32)
        img_16bit = np.round((img - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        img_pil = Image.fromarray(img_16bit, mode="I;16")
        img_pil.save(os.path.join(out_dir, f"{channel_name}_{i:05d}.png"))

# ------------------------------
#   Inference parameters
# ------------------------------
latent_dim = opt.latent_dim
channels = opt.channels
num_samples = opt.num_images
img_size = opt.resize_img
checkpoint_path = os.path.join('train_output', 'models', 'DCGAN',
                               opt.experiment_name, opt.model_G)
exp_name = opt.experiment_name

# ------------------------------
#   Load dataset to get min/max and original size
# ------------------------------
dataset_path = opt.dataset_dir
data_dict = torch.load(dataset_path)
outer_key = list(data_dict.keys())[opt.outer_key]
u = data_dict[outer_key]["u"]
v = data_dict[outer_key]["v"]
H_orig, W_orig = u.shape[0], u.shape[1]
u_min, u_max = float(u.min()), float(u.max())
v_min, v_max = float(v.min()), float(v.max())
print(f"Dataset stats: u_min={u_min}, u_max={u_max}, v_min={v_min}, v_max={v_max}, H_orig={H_orig}, W_orig={W_orig}")

# ------------------------------
#   Load generator
# ------------------------------
generator = Generator(latent_dim, img_size, channels).to(device)
state_g = torch.load(checkpoint_path)
generator.load_state_dict(state_g)
generator.eval()
print("Loaded generator from", checkpoint_path)

time_g, time_pp, time_sum = [], [], []

# ------------------------------
#   Generate samples
# ------------------------------

gen_imgs_list = []
time_g = []
mem_g = []

for i in range(num_samples):
    # --- CPU preprocessing ---
    z_cpu = torch.randn(1, latent_dim)  # CPU tensor
    
    # --- Move only for computation ---
    z = z_cpu.to(device)
    generator = generator.to(device)
    
    # Synchronize and reset peak memory stats
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- Measure time and memory for generator computation ---
    start_time_generator = time()
    
    with torch.no_grad():
        gen_img = generator(z)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time_generator = time()
    
    # Time measurement
    time_generator = end_time_generator - start_time_generator
    time_g.append(time_generator)
    
    # GPU memory measurement
    if device == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mem_g.append((mem_alloc, mem_peak))
    else:
        mem_g.append((0, 0))  # CPU fallback

    # --- Post-process the image ---
    gen_img_denorm = denormalize(gen_img, u_min, u_max, v_min, v_max, H_orig, W_orig)
    gen_imgs_list.append(gen_img_denorm.cpu())
    
    # Optionally release GPU memory
    del z, gen_img
    if device == "cuda":
        torch.cuda.empty_cache()

# --- Print summary ---
print('Average time to generate an image:', np.mean(time_g))
gen_imgs_denorm = torch.cat(gen_imgs_list, dim=0)  # (num_samples, 2, H, W)
print("Generated all images:", gen_imgs_denorm.shape)

# ------------------------------
#   Prepare output directories
# ------------------------------
u_all_dir = os.path.join("inference_output", exp_name, "u_channel")
v_all_dir = os.path.join("inference_output", exp_name, "v_channel")
os.makedirs(u_all_dir, exist_ok=True)
os.makedirs(v_all_dir, exist_ok=True)

# ------------------------------
#   Save ALL outputs: PNGs + one PT file
# ------------------------------
u_frames = []
v_frames = []
for i in range(num_samples):
    u_img = gen_imgs_denorm[i, 0].cpu().numpy().astype(np.float32)
    v_img = gen_imgs_denorm[i, 1].cpu().numpy().astype(np.float32)

    # Save PNGs (8-bit)
    plt.imsave(os.path.join(u_all_dir, f"img_{i:05d}.png"), u_img, cmap="gray")
    plt.imsave(os.path.join(v_all_dir, f"img_{i:05d}.png"), v_img, cmap="gray")
    # ------------------------------
    #   Convert to numpy arrays
    # ------------------------------
    u_array = gen_imgs_denorm[:, 0].numpy()
    v_array = gen_imgs_denorm[:, 1].numpy()

    # ------------------------------
    #   Prepare 16-bit output directories
    # ------------------------------
    u_out_dir = os.path.join("inference_output", exp_name, "u_channel_16bit")
    v_out_dir = os.path.join("inference_output", exp_name, "v_channel_16bit")

    # ------------------------------
    #   Save all frames as 16-bit PNGs using dataset min/max
    # ------------------------------
    save_uv_16bit_png(u_array, u_out_dir, "u", u_min, u_max)
    save_uv_16bit_png(v_array, v_out_dir, "v", v_min, v_max)

    # Collect arrays
    u_frames.append(u_img)
    v_frames.append(v_img)

# Stack arrays (H, W, T)
u_array = np.stack(u_frames, axis=-1)
v_array = np.stack(v_frames, axis=-1)

# Save one PT file
gen_data = {"u": torch.from_numpy(u_array), "v": torch.from_numpy(v_array)}
pt_path = os.path.join("inference_output", exp_name, "generated.pt")
torch.save(gen_data, pt_path)

print(f"Saved {num_samples} PNGs and one PT file with all arrays: {pt_path}")
