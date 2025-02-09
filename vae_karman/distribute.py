import os
import params
import util
import torch
import torch.distributed as dist
import torch.multiprocessing as mult
from itertools import chain


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialise the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def spawn_processes(fn):

    # Setup hyperparameters, initialise experiment
    args = params.get_args()
    experiment_path = util.initialise_experiment(args.experiment_name)

    n_gpus = torch.cuda.device_count()
    print(f'[{args.experiment_name}] Found {n_gpus} GPUs. Spawning processes...')

    # Spawn the processes
    mult.spawn(fn,
               args=(n_gpus, args, experiment_path),
               nprocs=n_gpus,
               join=True)


def cleanup():
    dist.destroy_process_group()


def load(experiment_path, args_model, ddp_model, optimizer=None):
    model_path = f'{experiment_path}/checkpoints/{args_model}'

    checkpoint = torch.load(model_path)
    epoch_start = checkpoint['epoch'] + 1
    ddp_model.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Make sure everyone is loaded before proceeding
    dist.barrier()
    return epoch_start


def save(experiment_path, epoch, ddp_model, optimizer):
    torch.save({
        'epoch': epoch,
        'model': ddp_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f'{experiment_path}/checkpoints/model_{epoch}.pt')


def interleave_arrays(*arrays):
    interleaved = list(chain.from_iterable(zip(*arrays)))
    return interleaved
