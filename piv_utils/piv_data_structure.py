import torch
import numpy as np
import pandas as pd

PT_PATH = "/path/to/data"   # <-- change me

# Load (map to CPU so it works even if tensors were saved from CUDA)
data = torch.load(PT_PATH, map_location="cpu")

rows = []

def add_row(path, obj):
    typ    = type(obj).__name__
    shape  = None
    dtype  = None
    device = None
    size   = None

    if torch.is_tensor(obj):
        typ    = "torch.Tensor"
        shape  = tuple(obj.shape)
        dtype  = str(obj.dtype)
        device = str(obj.device)
        size   = obj.numel()
    elif isinstance(obj, np.ndarray):
        typ   = "np.ndarray"
        shape = tuple(obj.shape)
        dtype = str(obj.dtype)
        size  = int(obj.size)
    elif isinstance(obj, (bytes, bytearray)):
        size  = len(obj)

    rows.append({
        "path": path,
        "type": typ,
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "size": size,
    })

def walk(obj, path=""):
    # Dicts
    if isinstance(obj, dict):
        add_row(path or "<root>", obj)
        for k, v in obj.items():
            walk(v, f"{path}/{k}" if path else str(k))
    # Lists/Tuples
    elif isinstance(obj, (list, tuple)):
        add_row(path, obj)
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")
    # Tensors/arrays/other leaves
    else:
        add_row(path, obj)

# Walk once
walk(data)

# Build a summary table
df = pd.DataFrame(rows)
# Sort by path for readability
df = df.sort_values("path", kind="mergesort").reset_index(drop=True)

# Print a compact view (first 100 rows) and save the full inventory
print(df.head(100).to_string(index=False))
df.to_csv("pt_inventory.csv", index=False)
print("\nSaved full inventory to pt_inventory.csv")
