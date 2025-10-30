import torch

import distribute
import time

from model import VAE
from datetime import datetime
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

import numpy as np
import matplotlib.pyplot as plt


def sample(rank, world_size, args, experiment_path):
    print(f'[{args.experiment_name}] [{rank}] Ready')

    # Initialise process group
    distribute.setup(rank, world_size)
    # Setup VAE model
    model = VAE(
        in_channels=1,  # Grayscale
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        kld_weight=args.kld_weight,
        auto_normalise=True
    ).to(rank)

    ddp_model = DistributedDataParallel(model, device_ids=[rank])

    epoch = distribute.load(experiment_path, args.model, ddp_model)
    print(f'[{args.experiment_name}] Loaded model {args.model} with {epoch} epoch(s)')

    start = time.time()
    with torch.inference_mode():
        runs = args.sample_size // args.batch_size
        remainder = args.sample_size % args.batch_size
        if remainder != 0:
            runs = runs + 1
        for i in range(runs):
            # Is this the run that accounts for the remainder?
            # This condition will never be hit if there is no remainder
            if i == runs:
                args.batch_size = remainder
            print(f'[{args.experiment_name}] Starting run [{i + 1}/{runs}] of size {args.batch_size}')
            sampled_images = ddp_model.module.sample(args.batch_size, rank)
            current_datetime = datetime.now()

            # Format the date and time for a filename (e.g., YYYY-MM-DD_HH-MM-SS)
            formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')

            for j in range(args.batch_size):
                sample_name = f'sample_{j}_{formatted_datetime}.png'
                img = sampled_images[j].detach().cpu().numpy()
                img = np.squeeze(img, axis=0)  # Remove channels dimension
                img = (img * 255).astype(np.uint8)  # Un-normalise and convert to int

                new_size = (850, 600)
                img_rescaled = Image.fromarray(img)  # Convert numpy array to PIL image
                img_rescaled = img_rescaled.resize(new_size)  # Resize the image
                img_rescaled = np.array(img_rescaled)

                # Add back in removed white columns
                new_img = np.full((img_rescaled.shape[0], img_rescaled.shape[1] + 150), 251, dtype=np.uint8)
                new_img[:, 150:] = img_rescaled  # Place the original image on the right side of the new array

                plt.imsave(f'{experiment_path}/runs/{args.model}/{sample_name}', new_img, cmap='gray')
                print(f'[{args.experiment_name}] Saved sample as {sample_name}')
    end = time.time()
    print(f'[{args.experiment_name}] Finished in {end - start} seconds')
    distribute.cleanup()


if __name__ == '__main__':
    distribute.spawn_processes(sample)
