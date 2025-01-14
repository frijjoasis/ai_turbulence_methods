import distribute
import util
import math
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.cuda.amp as amp
import torch.distributed as dist
import matplotlib.pyplot as plt
from model import VAE
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from torch.nn.parallel import DistributedDataParallel


def train(rank, world_size, args, experiment_path):
    print(f'[{args.experiment_name}] [{rank}] Ready')

    # Initialise process group
    distribute.setup(rank, world_size)

    # Setup VAE model
    model = VAE(
        in_channels=1,   # Grayscale
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        kld_weight=args.kld_weight,
        auto_normalise=True
    ).to(rank)

    preprocess = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                     transforms.Grayscale(),
                                     transforms.ToTensor()])

    train_set = datasets.ImageFolder(root=args.dataset_dir, transform=preprocess)

    # Create distributed sampler pinned to rank
    sampler = data.DistributedSampler(train_set,
                                      num_replicas=world_size,
                                      rank=rank,
                                      shuffle=True,
                                      seed=args.seed)

    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   sampler=sampler,
                                   pin_memory=True)

    ddp_model = DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        # util.show_random_datum(train_loader)
        total_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        print(f'[{args.experiment_name}] [{rank}] Model has {total_params} parameters')
        print(f'[{args.experiment_name}] [{rank}] Found {len(train_set)} images')

    scaler = amp.GradScaler()
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.learning_rate, betas=args.adam_betas)

    # Load model to continue training
    epoch_start = 0
    if args.model != "":
        epoch_start = distribute.load(experiment_path, args.model, ddp_model, optimizer=optimizer)

        print(f'[{args.experiment_name}] [{rank}] Successfully loaded model {args.model}')
        print(f'[{args.experiment_name}] [{rank}] Starting from epoch {epoch_start}...')

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=(epoch_start - 1))

    losses = []
    with tqdm(
            total=(args.num_epochs * len(train_loader) * world_size),
            unit='batch',
            disable=(rank != 0),
            initial=(epoch_start * len(train_loader) * world_size)
    ) as bar:

        ddp_model.train()
        for epoch in range(epoch_start, args.num_epochs):
            bar.set_description(f'[{args.experiment_name}] [{epoch}/{args.num_epochs}]')

            # Required to make shuffling work properly in the distributed case
            sampler.set_epoch(epoch)

            loss_curve = []
            total_loss = 0
            for j, (images, labels) in enumerate(train_loader):
                images = images.to(rank)

                # Use mixed precision
                with amp.autocast():
                    # Compute loss for minibatch
                    recons, mu, log_var = ddp_model(images)
                    loss = ddp_model.module.loss_function(recons, images, mu, log_var)
                    loss = loss / args.grad_accumulation

                    # Accumulate loss for minibatch
                    total_loss += loss.item()

                # Accumulate scaled gradients
                scaler.scale(loss).backward()

                if (j + 1) % args.grad_accumulation == 0:
                    # Update the weights every args.grad_accumulation batches
                    # Optimize and backwards pass
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    loss_curve.append(total_loss)
                    bar.set_postfix(loss=total_loss)

                    # Reset accumulated loss to 0
                    total_loss = 0

                bar.update(world_size)

            scheduler.step()

            # Save the model every args.save_every epoch, collect loss curves and plot
            print(f'[{args.experiment_name}] [{rank}] Entering end of epoch tasks...')
            print(f'[{args.experiment_name}] [{rank}] Gathering loss curves...')
            if rank == 0:
                if (epoch + 1) % args.save_every == 0:
                    distribute.save(experiment_path, epoch, ddp_model, optimizer)
                    print(f'[{args.experiment_name}] [{epoch}/{args.num_epochs}] Saved model')

                collected = [[] for _ in range(world_size)]
                dist.gather_object(loss_curve, object_gather_list=collected)

                interleaved = distribute.interleave_arrays(*collected)
                losses.append(interleaved)
                util.plot_loss_curve(interleaved, epoch, f'{experiment_path}/loss_curves/epoch_{epoch}.png')

                # Use the VAE model to generate a sample
                if args.sample:
                    with torch.inference_mode():
                        # Sample images directly from latent space
                        sampled_images = ddp_model.module.sample(args.sample_size, rank)
                        grid = utils.make_grid(sampled_images, nrow=math.ceil(math.sqrt(args.sample_size)), normalize=True)

                        grid_np = grid.permute(1, 2, 0).cpu().numpy()
                        plt.imsave(f'{experiment_path}/samples/sample_{epoch}.png', grid_np, cmap='gray')

                        print(f'[{args.experiment_name}] [{epoch}/{args.num_epochs}] Saved sample')
            else:
                dist.gather_object(loss_curve)
            print(f'[{args.experiment_name}] [{rank}] Finished for epoch {epoch}')
            dist.barrier()

    if rank == 0:
        util.plot_loss_curves(losses, f'{experiment_path}/loss_curves/all.png')

    distribute.cleanup()


if __name__ == "__main__":

    # Start the training function on each GPU
    distribute.spawn_processes(train)
