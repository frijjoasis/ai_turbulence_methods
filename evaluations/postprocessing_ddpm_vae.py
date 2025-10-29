import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

EPS = 1e-3  # tolerance for normalized images

# ------------------------------
# Utility functions
# ------------------------------

def load_normalized_image(path):
    """Load grayscale image normalized to [0,1] as torch.Tensor [H,W]."""
    img = Image.open(path).convert("L")
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img)  # [H,W]

def is_corrupt(img, tol=EPS):
    """Check if image has values outside [0-tol,1+tol]."""
    min_val, max_val = float(img.min()), float(img.max())
    return min_val < -tol or max_val > 1.0 + tol

def postprocess_channel(img, min_val, max_val, H_orig, W_orig):
    """
    Denormalize -> rotate 90° left -> resize -> vertical flip.
    """
    # Denormalize (map [0,1] -> [min_val, max_val])
    img_denorm = img * (max_val - min_val) + min_val  # [H,W]

    # Rotate 90° left (integer rotation)
    img_rot = torch.rot90(img_denorm, k=1, dims=(0, 1))

    # Resize to exact target size
    img_resized = TF.resize(img_rot.unsqueeze(0), (H_orig, W_orig), antialias=True).squeeze(0)

    # Vertical flip (flip top-bottom)
    img_flipped = torch.flip(img_resized, dims=[0])

    return img_flipped

def save_with_colorbar(img, path, title=None):
    """Save image with colorbar for debugging."""
    fig, ax = plt.subplots()
    im = ax.imshow(img.cpu().numpy(), cmap="gray")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def save_clean_image(img, path):
    """Save clean grayscale image (no labels/axis)."""
    arr = img.cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, arr, cmap="gray", vmin=arr.min(), vmax=arr.max())

def save_16bit_png(img, path, min_val, max_val):
    """Save image as 16-bit PNG preserving full float range."""
    arr = img.cpu().numpy().astype(np.float32)
    arr_16bit = np.round((arr - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_16bit, mode="I;16").save(path)

# ------------------------------
# Paths and dataset info
# ------------------------------

# CUSTOMIZE
# Set the paths, the outer key and the generative model
dataset_path = "/path/to/piv_data.pt'"
input_dir = "/path/to/data_to_be_processed.pt'"
outer_key_pos = 3
gen_model = "ddpm"

data_dict = torch.load(dataset_path)
outer_key = list(data_dict.keys())[outer_key_pos]

u = data_dict[outer_key]["u"]
v = data_dict[outer_key]["v"]
H_orig, W_orig = u.shape[0], u.shape[1]

u_min, u_max = float(u.min()), float(u.max())
v_min, v_max = float(v.min()), float(v.max())

file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

# ------------------------------
# Define output directories
# ------------------------------

debug_root = os.path.join("postprocessed_output_ddpm_vae", f"{gen_model}_{outer_key_pos}")
inference_root = os.path.join(f"{gen_model}_inference", f"{gen_model}_{outer_key_pos}")
pt_root = os.path.join(f"{gen_model}_inference", f"{gen_model}_{outer_key_pos}")

dirs = {
    "debug": {"u": os.path.join(debug_root, "u_channel"),
              "v": os.path.join(debug_root, "v_channel")},
    "inference": {"u": os.path.join(inference_root, "u_channel"),
                  "v": os.path.join(inference_root, "v_channel")},
    "16bit": {"u": os.path.join(inference_root, "u_channel_16bit"),
              "v": os.path.join(inference_root, "v_channel_16bit")},
}

# Create directories
for key in dirs:
    for channel in dirs[key]:
        os.makedirs(dirs[key][channel], exist_ok=True)

# ------------------------------
# Process & save images + arrays
# ------------------------------

u_list, v_list = [], []  # collect tensors for .pt file

# --- DEBUG: first 10 images with colorbar ---
for i, fname in enumerate(file_list[:10]):
    fpath = os.path.join(input_dir, fname)
    img = load_normalized_image(fpath)
    if is_corrupt(img):
        print(f"Skipping {fname}: out of range")
        continue

    if not 'average' in fname.lower():
        if "u" in fname.lower():
            img_proc = postprocess_channel(img, u_min, u_max, H_orig, W_orig)
            channel_name = "u"
        elif "v" in fname.lower():
            img_proc = postprocess_channel(img, v_min, v_max, H_orig, W_orig)
            channel_name = "v"
        else:
            continue
    else:
        continue

    debug_path = os.path.join(dirs["debug"][channel_name], f"sample_{i}.png")
    save_with_colorbar(img_proc, debug_path, title=f"{channel_name}-channel")

print("Saved first 10 debug images with colorbars.")

# --- FULL INFERENCE ---
for i, fname in enumerate(file_list):
    fpath = os.path.join(input_dir, fname)
    img = load_normalized_image(fpath)
    if is_corrupt(img):
        print(f"Skipping {fname}: out of range")
        continue
    
    if not 'average' in fname.lower():
        if "u" in fname.lower():
            img_proc = postprocess_channel(img, u_min, u_max, H_orig, W_orig)
            channel_name = "u"
            u_list.append(img_proc.unsqueeze(-1))  # add time dimension
            # Save 16-bit PNG
            save_16bit_png(img_proc, os.path.join(dirs["16bit"]["u"], fname), u_min, u_max)
        elif "v" in fname.lower():
            img_proc = postprocess_channel(img, v_min, v_max, H_orig, W_orig)
            channel_name = "v"
            v_list.append(img_proc.unsqueeze(-1))
            # Save 16-bit PNG
            save_16bit_png(img_proc, os.path.join(dirs["16bit"]["v"], fname), v_min, v_max)
        else:
            continue
    else:
        continue

    # Save clean visualization PNG
    inf_path = os.path.join(dirs["inference"][channel_name], fname)
    save_clean_image(img_proc, inf_path)

    if i % 100 == 0:
        print(f'Processed {i} frames')

# Stack into (H, W, T) and save .pt file
if u_list and v_list:
    u_array = torch.cat(u_list, dim=-1)
    v_array = torch.cat(v_list, dim=-1)
    save_dict = {"u": u_array, "v": v_array}

    pt_out_path = os.path.join(pt_root, "generated.pt")
    torch.save(save_dict, pt_out_path)
    print(f"Saved denormalized arrays as {pt_out_path}")

print("Postprocessing completed: PNGs + 16-bit PNGs + .pt arrays saved.")
