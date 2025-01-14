import torch
from torch import nn
from torch.nn import functional as F


def reparameterise(mu, log_var):
    # Reparameterise trick for sampling the latent variable
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def normalise_data(img):
    return img * 2 - 1


def unnormalise_data(t):
    return (t + 1) * 0.5


class VAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims,
                 kld_weight,
                 auto_normalise=False,
                 ):
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.kld_weight = kld_weight

        modules = []

        # Build Encoder
        in_size = in_channels
        for i in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_size, out_channels=i, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(i),
                    nn.LeakyReLU())
            )
            in_size = i

        self.encoder = nn.Sequential(*modules)

        # Note that this may not be the correct dimensionality, depending on the image size and hidden_dims
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            # Also may not be the correct output dimensionality, depending on image size
            nn.Conv2d(hidden_dims[-1], out_channels=in_channels, kernel_size=3, padding=1),
            nn.Tanh())

        self.normalise = normalise_data if auto_normalise else lambda x: x
        self.unnormalise = unnormalise_data if auto_normalise else lambda x: x

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x):
        x = self.normalise(x)
        mu, log_var = self.encode(x)
        z = reparameterise(mu, log_var)
        recons = self.decode(z)
        return recons, mu, log_var

    def loss_function(self, recons, x, mu, log_var):
        # Reconstruction term loss
        recons_loss = F.mse_loss(recons, x)

        # KL term loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        return loss

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)

        samples = self.decode(z)
        samples = self.unnormalise(samples)
        return samples
