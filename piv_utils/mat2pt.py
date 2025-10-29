import torch
import numpy as np
from scipy.io import loadmat
import argparse

def convert_mat_to_pt(mat_path, pt_path):
    """
    Load a .mat file, extract all numeric data, and save it as a .pt file.
    """
    def extract_data(obj):
        """Recursively extract numeric arrays from MATLAB structs/cells."""
        if isinstance(obj, np.ndarray):
            # Handle numeric arrays
            if obj.dtype.kind in {'i', 'f'}:
                return torch.tensor(obj, dtype=torch.float32)
            # Handle object arrays (cells or structs)
            elif obj.dtype == np.object_:
                extracted = []
                for item in obj.flatten():
                    extracted.append(extract_data(item))
                return extracted
        elif isinstance(obj, dict):
            return {k: extract_data(v) for k, v in obj.items() if not k.startswith('__')}
        return obj  # Leave non-numeric or unsupported types as-is

    print(f"Loading {mat_path} ...")
    data = loadmat(mat_path, simplify_cells=False)
    extracted_data = {k: extract_data(v) for k, v in data.items() if not k.startswith('__')}

    print(f"Saving extracted data to {pt_path} ...")
    torch.save(extracted_data, pt_path)
    print("Conversion completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mat to .pt (PyTorch) format.")
    parser.add_argument("--input", required=True, help="Path to the .mat file (CylinderArrays.mat)")
    parser.add_argument("--output", default="output.pt", help="Path to save the .pt file")
    args = parser.parse_args()

    convert_mat_to_pt(args.input, args.output)
