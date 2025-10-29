"""
Code to realize the inference for DCGAN models
Implementation by Claudia Drygala, University of Wuppertal
"""

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
import numpy as np
from numpy import linspace, asarray
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os
from options.inference_options import InferenceOptions
from math import sqrt
from scipy.stats import norm
from pylab import plot, show, grid, xlabel, ylabel
from time import time

# We always train on a gpu!
cuda = True

# Parse the inference options
opt = InferenceOptions().parse()

# Define the gpu the code should be performed on
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

# Set the image size, since we need it multiple times in the code
img_size = opt.resize_img

# Define at the beginning if seed should be set
if opt.seed:
    torch.manual_seed(0)

# Naive approach: Generate random noise vector sampled from a normal distribution
def noise_std_normal(size):
    n = Variable(torch.randn(size, opt.latent_dim))
    return n.cuda()

# Bilinear interpolation: Uniform interpolation between two points in latent space
def interpolate_points(p1, p2, inter_steps):
    # interpolate ratios between the points
    ratios = linspace(0, 1, num=inter_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return vectors

# Wiener process to define latent vectors
# Implementation derived by: https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html

def brownian_motion_1D(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)
    out = torch.from_numpy(out).float()
    out = Variable(out)
    return out.cuda()

# Ornstein_Uhlenbeck process to define latent vectors
# Implementations follows https://www.pik-potsdam.de/members/franke/lecture-sose-2016/introduction-to-python.pdf
def ou(y0, n, t, dt, theta, mu, sigma, out=None):
    y0 = np.asarray(y0)
    drift = lambda y0,t: theta*(mu-y0)      # define drift term
    diffusion = lambda y0,t: sigma # define diffusion term

    r = norm.rvs(size=y0.shape,loc=0.0,scale=1.0) * np.sqrt(dt) #define noise process
    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # solve SDE
    for i in range(2,n):
        y0[:,i] = y0[:,i-1] + drift(y0[:,i-1],i*dt)*dt + diffusion(y0[:,i-1],i*dt)*r[:,i]

    print(y0.shape)
    out = y0
    # Add the initial condition.
    # out += np.expand_dims(y0, axis=-1)
    out = torch.from_numpy(out).float()
    out = Variable(out)
    return out.cuda()


# Convert output of generator to image
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, img_size, img_size)

# Definition of the generator
# Generators input is a latent variable vector
# Output: A n_out valued vector which corresponds to the flattened vector of the given images
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Initialize G
generator = Generator()
generator.cuda()

# Load the weights of the model
state_g = torch.load(os.path.join(opt.checkpoint_dir, opt.model_G))
generator.load_state_dict(state_g)

# Compute the latent vectors of which the images are generated
# The naive approach deliver vectors which are completely randomly sampled.
# By the bilinear approach it is possible to get sequences with fluent transitions between each image,
# bilinear2P is a special case where the images of the sequence will skip only between two noises, so one reproce a reptetive sequence.

noise_list = []
if opt.latent_type == 'naive':
    noise_list = noise_std_normal(size=opt.num_images)
elif opt.latent_type == 'bilinear':
    latent_vector_num = int(opt.num_images / opt.bi_steps) + 1
    g_noise =  noise_std_normal(size=latent_vector_num)
    for vec in range(len(g_noise)-1):
        inter_noise = interpolate_points(g_noise[vec], g_noise[vec+1], inter_steps=opt.bi_steps)
        noise_list.extend(inter_noise)
elif opt.latent_type == 'bilinear2P':
    g_noise = noise_std_normal(2)
    latent_vector_num = int(opt.num_images / opt.bi_steps) + 1
    for vec in range(latent_vector_num):
        if vec % 2 == 0:
            inter_noise = interpolate_points(g_noise[0], g_noise[1], inter_steps=opt.bi_steps)
        else:
            inter_noise = interpolate_points(g_noise[1], g_noise[0], inter_steps=opt.bi_steps)
        noise_list.extend(inter_noise)
elif opt.latent_type == 'bm':
    # The parameters for Brownian motion. Choosen in this way, that the random variables are distributed to N(0,1)
    delta = 1.0
    T = 1.0 # Total time
    N = opt.latent_dim # Number of steps
    dt = T / N   # Time step size
    m = opt.num_images # Number of realizations to generate
    x = np.empty((m, N + 1)) # Create an empty array to store the realizations
    x[:, 0] =  np.random.normal(loc=0.0,scale=1.0) # Initial values of x
    noise_list = brownian_motion_1D(x[:, 0], N, dt, delta, out=x[:, 1:])
elif opt.latent_type == 'ou':
    # The parameters for the Ornstein-Uhlenbeck Process. Choosen in this way, that the random variables are distributed to N(0,1).
    t_0 = 0
    t_end = 1
    time_steps = opt.latent_dim
    theta = 1.1
    mu = 0.3
    sigma = 0.8
    t = np.linspace(t_0, t_end,time_steps)  # define time axis
    dt = np.mean(np.diff(t))
    m = opt.num_images # Number of realizations to generate
    y = np.empty((m, time_steps)) # Create an empty array to store the realizations
    y[:,0] = np.random.normal(loc=0.0,scale=1.0)  # initial condition
    noise_list = ou(y, time_steps, t, dt, theta, mu, sigma, out=y[:,1:])

else:
    raise ValueError('Sorry, the type for computation of the latent vectors is not known! Please choose one of the possible options: naive, bilinear, bilinear2P, random_walk')

# Make directory to save the images
save_dir = os.path.join('../inference_output', opt.experiment_name, opt.latent_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

time_g = []
time_pp = []
time_sum = []
print('Generated image...')
for count in range(opt.num_images):
    test_noise = torch.reshape(noise_list[count], (1, opt.latent_dim))
    start_time_generator = time()
    test_images = vectors_to_images(generator(test_noise))
    end_time_generator = time()
    time_generator = end_time_generator - start_time_generator
    time_g.append(time_generator)
    test_images = test_images.data
    test_images = test_images.cpu().numpy()
    test_images = test_images.reshape(img_size, img_size)
    if opt.image_type == 'karman':
        start_time_image_processing = time()
        fig = plt.figure(figsize=(10,6))
        plt.axis('off')
        plt.imshow(test_images)
        fig.savefig(os.path.join(save_dir, str(count)), bbox_inches='tight', transparent= True, pad_inches = 0, dpi=130)
        plt.close()
        img = PIL.Image.open(os.path.join(save_dir, str(count))+'.png')
        gray_img = img.convert("L")
        gray_img = gray_img.resize((1000,600), Image.ANTIALIAS)
        fig = plt.figure(figsize=(10,6))
        plt.axis('off')
        plt.imshow(gray_img, cmap='gray')
        fig.savefig(os.path.join(save_dir, str(count)), bbox_inches='tight', transparent= True, pad_inches = 0, dpi=130)
        plt.close()
        end_time_image_processing = time()
        time_image_processing = end_time_image_processing - start_time_image_processing
        time_pp.append(time_image_processing)
        time_image_desired = time_generator + time_image_processing
        time_sum.append(time_image_desired)
        print('Time to generate an image:', time_generator)
        print('Time for postprocessing:', time_image_processing)
        print('Time to derive desired image:', time_image_desired)
    else:
        fig = plt.figure(figsize=(10, 6.25))
        plt.axis('off')
        plt.imshow(test_images)
        fig.savefig(os.path.join(save_dir, str(count)), bbox_inches='tight', transparent=True, pad_inches=0, dpi=130)
        plt.close()
        img = PIL.Image.open(os.path.join(save_dir, str(count)) + '.png')
        gray_img = img.convert("L")
        gray_img = gray_img.resize((1000, 625), Image.ANTIALIAS)
        fig = plt.figure(figsize=(10, 6.25))
        plt.axis('off')
        plt.imshow(gray_img, cmap='gray')
        fig.savefig(os.path.join(save_dir, str(count)), bbox_inches='tight', transparent=True, pad_inches=0, dpi=130)
        plt.close()
    print('...', count)
print('Finished!')

print('-----Time on average-----')
print('Time to generate an image:', np.mean(time_g))
print('Time for postprocessing:', np.mean(time_pp))
print('Time to derive desired image:', np.mean(time_sum))
