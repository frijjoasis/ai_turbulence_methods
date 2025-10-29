import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# --------------------------------------------------------
# Helpers - Ensurance data is not corrupted after loading
# --------------------------------------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.array(x)

def compute_global_stats(u, v):
    u_np = to_numpy(u).ravel()
    v_np = to_numpy(v).ravel()
    combined = np.concatenate([u_np, v_np])
    return float(combined.min()), float(combined.max())

def ensure_orientation(img, target_h, target_w):
    """Ensure array is (target_h, target_w). If transposed, fix with .T"""
    h, w = img.shape
    if (h, w) == (target_h, target_w):
        return img
    elif (w, h) == (target_h, target_w):
        return img.T
    else:
        raise ValueError(f"Unexpected shape {img.shape}, expected ({target_h},{target_w}) or ({target_w},{target_h})")

def align_channel_pairs(u_arr, v_arr, name=''):
    if u_arr.shape[0] != v_arr.shape[0]:
        n = min(u_arr.shape[0], v_arr.shape[0])
        print(f"Warning: {name} mismatch u={u_arr.shape[0]}, v={v_arr.shape[0]} → trimming to {n}")
        u_arr, v_arr = u_arr[:n], v_arr[:n]
    return u_arr, v_arr


# --------------------------------------------------------
# Data loader
# --------------------------------------------------------
def load_channel_pt(pt_path, channel="u", target_size=None, revert=False, vflip=True, max_frames=3000, outer_key_idx=None):
    """
    Load channel data ('u' or 'v') from a .pt file.
    Works for both raw dataset files (with outer_keys) and generated inference files.

    Args:
        pt_path (str): Path to .pt file
        channel (str): "u" or "v"
        target_size (tuple): (H, W) to enforce orientation
        revert (bool): If True, apply intensity reversal (img.max() - img) per frame
        max_frames (int): Max number of frames to load
        outer_key_idx (int or None): Which outer_key to use (only needed for raw dataset .pt)

    Returns:
        np.ndarray of shape (N, H, W)
    """
    data = torch.load(pt_path, map_location="cpu")

    # Case 1: raw dataset with outer_keys
    if isinstance(list(data.values())[0], dict):
        if outer_key_idx is None:
            raise ValueError("outer_key_idx must be provided for raw dataset .pt")
        outer_keys = list(data.keys())
        outer_key = outer_keys[outer_key_idx]
        data = data[outer_key]

    if channel not in data:
        raise KeyError(f"Channel '{channel}' not found in {pt_path}. Available: {list(data.keys())}")

    arr = data[channel].numpy().astype(np.float32)  # (H, W, T)
    H, W, T = arr.shape
    n = min(T, max_frames)

    imgs = []
    for t in range(n):
        img = arr[:, :, t]

        if target_size is not None:
            img = ensure_orientation(img, *target_size)

        if revert:
            img = img.max() - img
        
        if vflip:
            img = np.flipud(img)  # vertical flip

        imgs.append(img)

    return np.stack(imgs, axis=0)  # (N,H,W)

# -----------------------------
# Statistics
# -----------------------------
def combine_channels(u_arr, v_arr, mode='magnitude'):
    """
    Combine u and v channels into a scalar or vector representation.

    Args:
        u_arr, v_arr (np.ndarray): arrays of same shape
        mode (str): how to combine channels
            - 'u_only'     -> return u component
            - 'v_only'     -> return v component
            - 'magnitude'  -> return sqrt(u^2 + v^2)
            - 'vector'     -> return stacked [u, v] along last axis
    """
    if mode == 'u_only':
        return u_arr
    elif mode == 'v_only':
        return v_arr
    elif mode == 'magnitude':
        return np.sqrt(u_arr**2 + v_arr**2)
    elif mode == 'vector':
        return np.stack([u_arr, v_arr], axis=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def computation_mean(img_array, x_range, y_start=10, y_end=220,
                     normalize="independent", global_scale=None):
    """
    Compute mean profile along y.
    """
    N, H, W = img_array.shape
    y_end = min(y_end, H)

    # mean across all frames and x_range
    mean_profile = np.mean(img_array, axis=0)  # (H, W)
    curve = np.mean(mean_profile[y_start:y_end, 0:x_range], axis=1)

    scale = 1.0
    if normalize == "independent":
        max_val = np.max(np.abs(curve))
        if max_val > 0:
            curve = curve / max_val
            scale = max_val
    elif normalize == "global":
        if global_scale is None:
            raise ValueError("global_scale must be provided for global normalization")
        curve = curve / global_scale
        scale = global_scale

    return curve, scale


def computation_var(img_array, x_range, y_start=10, y_end=220,
                    normalize=True, normalization_scale=None, ddof=0):
    """
    Compute variance profile along y.
    Uses normalization_scale from mean (squared) for consistent scaling.
    """
    N, H, W = img_array.shape
    y_end = min(y_end, H)
    var_profile = np.var(img_array, axis=0, ddof=ddof)
    curve = np.mean(var_profile[y_start:y_end, 0:x_range], axis=1)

    if normalize:
        if normalization_scale is None:
            max_val = np.max(np.abs(curve))
            if max_val > 0:
                curve = curve / max_val
        else:
            curve = curve / (normalization_scale**2)
    return curve


def compute_global_magnitude_scale(u_arr, v_arr):
    """
    Compute global max magnitude across dataset for normalization.
    """
    magnitude = np.sqrt(u_arr**2 + v_arr**2)
    return np.max(magnitude)
    

def compute_global_magnitude_scale_for_sets(list_of_u, list_of_v):
    """
    Compute a single global magnitude scale across multiple datasets.

    Args:
        list_of_u (list of np.ndarray): each element shape (N,H,W)
        list_of_v (list of np.ndarray): same length as list_of_u

    Returns:
        global_mag_scale (float): max magnitude across all frames/pixels/datasets
        comp_scales (dict): {'u': max_abs_u, 'v': max_abs_v}
    """
    if len(list_of_u) != len(list_of_v):
        raise ValueError("list_of_u and list_of_v must have same length")

    max_mag = 0.0
    max_u = 0.0
    max_v = 0.0

    for u_arr, v_arr in zip(list_of_u, list_of_v):
        # ensure numpy
        u = to_numpy(u_arr)
        v = to_numpy(v_arr)
        max_u = max(max_u, np.max(np.abs(u)))
        max_v = max(max_v, np.max(np.abs(v)))

        # compute magnitude per-dataset
        mag = np.sqrt(u**2 + v**2)
        max_mag = max(max_mag, np.max(mag))

    comp_scales = {'u': float(max_u), 'v': float(max_v)}
    return float(max_mag), comp_scales

# -----------------------------
# Plotting
# -----------------------------
def resample_profile(profile, target_len):
    orig_len = profile.shape[0]
    if orig_len == target_len:
        return profile
    x_orig = np.linspace(0, 1, orig_len)
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_orig, profile)


def plot_metric(v_val_list, save_dir, name, labels=['Real', 'GAN', 'VAE', 'DDPM'], revert_curve=False):
    lengths = [p.shape[0] for p in v_val_list]
    if len(set(lengths)) != 1:
        target_len = max(lengths)
        v_val_list = [resample_profile(p, target_len) for p in v_val_list]
    else:
        target_len = lengths[0]

    y_delta = np.linspace(0, 1, target_len)
    cmap = plt.get_cmap('tab10')
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

    plt.figure()
    for i, prof in enumerate(v_val_list):
        if revert_curve:
            prof = -prof  # reverse along y-axis
        plt.plot(y_delta, prof, label=labels[i], linestyle=linestyles[i], color=cmap(i))
    # Legend outside, horizontal line, with box
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),       # center it above the plot
        ncol=len(labels),                 # all entries in one row
        frameon=True,                     # box around legend
        fancybox=True,                     # rounded box edges
        shadow=False                       # no shadow, clean look
    )

    plt.xlabel(r'$y/\delta$')
    plt.ylabel(r'$\psi$')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, name + ".jpeg"), dpi=200, bbox_inches='tight')
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Customize: Dataset, fluid ranges and channel modes for statistics
    outer_key_pos = 3
    # Outer key codes for dataset:
    # 0 - 2V2H
    # 1 - 4V2H
    # 3 - 2V8H
    # 4 - 4V8H
    fluid_ranges = [35, 70, 105, 137] 
    modes = ['magnitude']
    normalization_ref = 'sim' # mix or sim

    # Set datapaths
    les_pt_dir = "/path/to/piv_data.pt'" 
    gan_pt_dir = "/path/to/gan_data.pt"
    ddpm_pt_dir = "/path/to/ddpm_data.pt"
    vae_pt_dir = "/path/to/vae_data.pt"

    # Load simulated data
    les_u = load_channel_pt(les_pt_dir, channel="u", outer_key_idx=outer_key_pos)
    les_v = load_channel_pt(les_pt_dir, channel="v", outer_key_idx=outer_key_pos)
  
    # Load generated data
    gan_u = load_channel_pt(gan_pt_dir, channel="u")
    gan_v = load_channel_pt(gan_pt_dir, channel="v")

    ddpm_u = load_channel_pt(ddpm_pt_dir, channel="u")
    ddpm_v = load_channel_pt(ddpm_pt_dir, channel="v")

    vae_u = load_channel_pt(vae_pt_dir, channel="u")
    vae_v = load_channel_pt(vae_pt_dir, channel="v")

    # Check statistics
    print(f"Min and Max Value of LES (u channel):{les_u.min()} - {les_u.max()}")
    print(f"Min and Max Value of GAN (u channel):{gan_u.min()} - {gan_u.max()}")
    print(f"Min and Max Value of DDPM (u channel):{ddpm_u.min()} - {ddpm_u.max()}")
    print(f"Min and Max Value of VAE (u channel):{vae_u.min()} - {vae_u.max()}")

    # Compute global normalization scale and set save directory
    if normalization_ref == 'sim':
        global_scale = compute_global_magnitude_scale(les_u, les_v)
        # Set path to save evaluations
        save_dir = f"evaluation/global_scale_sim_new/data_{outer_key_pos}" 
    elif normalization_ref == 'mix':
        datasets_u = [les_u, gan_u, vae_u, ddpm_u]
        datasets_v = [les_v, gan_v, vae_v, ddpm_v]
        global_scale, _ = compute_global_magnitude_scale_for_sets(datasets_u, datasets_v)
        # Set path to save evaluations
        save_dir = f"evaluation/global_scale_mix/data_{outer_key_pos}" 
    else:
        raise ValueError(f'Global scale {normalization_ref} for normalization unknown.')


    for fluid_range in fluid_ranges:
        for mode in modes:
            # Combine u and v or just consider one channel
            les_comb = combine_channels(les_u, les_v, mode)
            gan_comb = combine_channels(gan_u, gan_v, mode)
            vae_comb = combine_channels(vae_u, vae_v, mode)
            ddpm_comb = combine_channels(ddpm_u, ddpm_v, mode)

            # Mean + scale (consistent global normalization)
            mean_les, S_les = computation_mean(les_comb, fluid_range,
                                               normalize="global", global_scale=global_scale)
            mean_gan, S_gan = computation_mean(gan_comb, fluid_range,
                                               normalize="global", global_scale=global_scale)
            mean_vae, S_vae = computation_mean(vae_comb, fluid_range,
                                               normalize="global", global_scale=global_scale)
            mean_ddpm, S_ddpm = computation_mean(ddpm_comb, fluid_range,
                                                 normalize="global", global_scale=global_scale)

            # Var (scaled consistently with squared mean)
            var_les = computation_var(les_comb, fluid_range, normalization_scale=global_scale)
            var_gan = computation_var(gan_comb, fluid_range, normalization_scale=global_scale)
            var_vae = computation_var(vae_comb, fluid_range, normalization_scale=global_scale)
            var_ddpm = computation_var(ddpm_comb, fluid_range, normalization_scale=global_scale)

            # Plot
            plot_metric([mean_les, mean_gan, mean_vae, mean_ddpm],
                        save_dir, f"mean_{mode}_{fluid_range}")
            plot_metric([var_les, var_gan, var_vae, var_ddpm],
                        save_dir, f"var_{mode}_{fluid_range}")

    print("Done.")