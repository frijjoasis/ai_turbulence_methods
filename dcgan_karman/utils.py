"""
Code for logging of the progress, extracted from the repo: https://github.com/diegoalejogm/gans

Modified by Claudia Drygala, University of Wuppertal
If desired, TensorBoard Data will be stored in '..train_output/runs' path
"""

import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from options.train_options import TrainOptions

# Parse the training options
opt = TrainOptions().parse()

class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # Initialize TensorBoard
        if opt.tensorboard_active:
            self.writer = SummaryWriter('../train_output/runs/{}'.format(self.data_subdir), comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        if opt.tensorboard_active:
            self.writer.add_scalar('{}/D_error'.format(self.comment), d_error, step)
            self.writer.add_scalar('{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        if format == 'NHWC':
            images = images.transpose(1, 3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/train_output/images_progress{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        if opt.tensorboard_active:
            self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=False):
        out_dir = '../train_output/images_progress/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        #fig = plt.figure()
        #plt.imshow(np.moveaxis(horizontal_grid.cpu().numpy(), 0, -1))
        #plt.axis('off')
        #if plot_horizontal:
        #    display.display(plt.gcf())
        #self._save_images(fig, epoch, n_batch, 'hori')
        #plt.close()

        # Plot and save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.cpu().numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = '../train_output/images_progress/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))


    def save_models(self, generator, discriminator, epoch):
        out_dir = '../train_output/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise