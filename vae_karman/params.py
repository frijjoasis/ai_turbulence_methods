import argparse
import json

with open('params.json', 'r') as file:
    params = json.load(file)


def get_args(sample=False):
    parser = argparse.ArgumentParser()

    # Torch settings
    parser.add_argument('--image_size', type=int, default=params["image_size"],
                        help='Side length (in pixels) training images will be resized to')
    parser.add_argument('--batch_size', type=int, default=params["batch_size"], help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=params["num_epochs"],
                        help='Number of times to loop through the training data')
    parser.add_argument('--dataset_dir', type=str, default=params["dataset_dir"],
                        help='Location of the training images')
    parser.add_argument('--grad_accumulation', type=int, default=params["grad_accumulation"],
                        help='Number of batches to accumulate before updating the model weights')
    parser.add_argument('--seed', type=int, default=params["seed"], help='Seed for shuffling the dataloader')

    # VAE settings
    parser.add_argument('--latent_dim', type=int, default=params["latent_dim"],
                        help='Dimension of the latent space')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=params["hidden_dims"],
                        help='Dimensions of the hidden layers')
    parser.add_argument('--kld_weight', type=float, default=params["kld_weight"],
                        help='Weight of the Kullback-Leibler divergence term in the loss')

    # Adam settings
    parser.add_argument('--learning_rate', type=float, default=params["learning_rate"],
                        help='Learning rate for the optimizer')
    parser.add_argument('--adam_betas', type=tuple, default=tuple(params["adam_betas"]),
                        help='Betas for the Adam optimizer')

    # Scheduler settings
    parser.add_argument('--gamma', type=float, default=params["gamma"],
                        help='The factor by which the learning rate is updated per epoch')

    # Inference settings
    parser.add_argument('--sample', type=bool, default=params["sample"],
                        help='Whether to make samples every "--save-every" epochs')
    parser.add_argument('--sample_size', type=int, default=params["sample_size"],
                        help='How many samples to produce; must be a square number')
    parser.add_argument('--model', type=str, required=sample, default=params["model"],
                        help='The model to load for training or inference, located in ./$experiment_name/checkpoints/')

    # Miscellaneous settings
    parser.add_argument('--experiment_name', type=str, required=True, default=params["experiment_name"],
                        help='Name of this run')
    parser.add_argument('--save_every', type=int, default=params["save_every"], help='How often to save the model')

    args = parser.parse_args()
    return args
