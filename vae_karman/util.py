import matplotlib.pyplot as plt
import shutil
import os
import torch


def show_random_datum(train_loader):
    # Get a random batch
    data_iterator = iter(train_loader)
    images, labels = next(data_iterator)

    # Choose a random index within the batch
    random_index = torch.randint(0, len(images), (1,)).item()

    # Extract the random image
    random_image = images[random_index].permute(1, 2, 0).numpy()

    # Display the random image
    plt.imshow(random_image, cmap='gray')
    plt.title(f"Random Image from dataset (Index: {random_index})")
    plt.axis('off')
    plt.show()


def initialise_experiment(experiment_name):
    # British spelling.

    folders = ['samples', 'checkpoints', 'loss_curves']
    experiment_path = os.path.join('experiments', experiment_name)

    # Create experiment folders if they don't exist
    for f in folders:
        os.makedirs(os.path.join(experiment_path, f), exist_ok=True)

    # Make a copy of this run's hyperparameters
    params_json_path = os.path.join(experiment_path, 'params.json')
    shutil.copy('params.json', params_json_path)

    return experiment_path


def plot_loss_curve(data, epoch, path):
    plt.figure()
    plt.plot(data, label=f'Epoch {epoch}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss against local step in epoch {epoch}')
    plt.savefig(path)
    plt.close()


def plot_loss_curves(data, path):
    plt.figure()
    for i, curve in enumerate(data):
        plt.plot(curve, label=f'Epoch {i}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss against local step in each epoch')
    plt.legend()
    plt.savefig(path)
    plt.close()
