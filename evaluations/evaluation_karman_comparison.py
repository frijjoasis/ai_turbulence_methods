# Evaluations published in ... (Drygala, Ross, Di Mare, Gottschalk)

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import scipy.stats


def load_data(data_dir):
    eliminated = 0
    imgs = []
    for root, _, files in os.walk(data_dir):
        for file in files: 
            file_dir = os.path.join(root, file)
            img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
            img_white = np.zeros(img.shape,dtype=np.uint8)
            if not img.min() > 0:
                img_white.fill(img.max())
                img_revert = np.subtract(img_white, img)
                img_normalized = np.divide(img_revert, img_white)
                imgs.append(img_normalized)
            else: 
                eliminated +=1
    print('Eliminated Images', eliminated)
    imgs = np.array(imgs)

    return imgs


def computation_mean_v(img_array, x_range):

    v_mean = np.mean(img_array, axis=0) 
    v_mean_range = np.mean(v_mean[:, 287:x_range], axis=1) # img[height, width]

    return v_mean_range


def computation_var_v(img_array, x_range):

    v_var = np.var(img_array, axis=0) 
    v_var_mean = np.mean(v_var[:, 287:x_range], axis=1) # img[height, width]
    
    return v_var_mean


def computation_square_v(img_array, x_range):

    v_sq = np.square(img_array) 
    v_sq_mean = np.mean(v_sq, axis=0)
    v_sq_range = np.mean(v_sq_mean[:, 287:x_range], axis=1) # img[height, width]
    
    return v_sq_range


def computation_std_dev_v(img_array, x_range):

    v_std = np.std(img_array, axis=0) 
    v_std_mean = np.mean(v_std[:, 287:x_range], axis=1) # img[height, width]
    
    return v_std_mean


def compute_confidence_interval(metric_vector, std_vector, number_test_images, alpha=0.05, mean=True):
    if mean: 
        t_norm = scipy.stats.t.ppf(1-alpha/2, number_test_images-1)
        upper_bound = metric_vector + t_norm*(1/np.sqrt(number_test_images))*std_vector
        lower_bound = metric_vector - t_norm*(1/np.sqrt(number_test_images))*std_vector
    else: 
        upper_bound =  ((number_test_images-1)*metric_vector) /scipy.stats.chi2.ppf(1-alpha/2, number_test_images-1)
        lower_bound =  ((number_test_images-1)*metric_vector) /scipy.stats.chi2.ppf(alpha/2, number_test_images-1)

    return upper_bound, lower_bound 


def plot_metric_v(v_val, save_dir, upper_bounds=None, lower_bounds=None, name='mean_v'):

    # v_val, upper_bounds and lower_bounds are lists with the computed values for les, gan, vae and ddpm

    img_height = v_val[0].shape[0]   # Parameter for normailizing y axis, delta_v_les.shape = delta_v_gan.shape

    # Color palettes: https://matplotlib.org/stable/users/explain/colors/colormaps.html

    cmap = plt.get_cmap('tab10')
    cles = cmap(1)
    cgan = cmap(0)
    cvae = cmap(2)
    cddpm = cmap(3)

    y_delta = np.arange(start=0, stop=img_height, step=1)/(img_height-1)

    plt.plot(y_delta, v_val[0], label='LES', color=cles)
    plt.plot(y_delta, v_val[1], label='GAN', color=cgan, linestyle='dashed')
    plt.plot(y_delta, v_val[2], label='VAE;', color=cvae, linestyle='dashdot')
    plt.plot(y_delta, v_val[3], label='DDPM;', color=cddpm, linestyle='dotted')
    
    plt.legend()
    plt.xlabel(r'$y/\delta$')
    plt.ylabel(r'$\psi$')
    plt.savefig(os.path.join(save_dir, name + '.jpeg'))
    plt.close()


if __name__ == "__main__":
    # Investigated range of fluid (equals x_range of image)

    fluid_ranges = [300, 500]

    # Set the data paths
    DATA_PATH_LES = os.path.join('/path/to/les_data')
    DATA_PATH_GAN = os.path.join('/path/to/gan_data') 
    DATA_PATH_DDPM = os.path.join('/path/to/ddpm_data') 
    DATA_PATH_VAE = os.path.join('/path/to/vae_data') 
    DATA_PATH_SAVE_EVALUATION = os.path.join('/path/to/save_eval', 'folder_name')

    if not os.path.exists(DATA_PATH_SAVE_EVALUATION):
        os.makedirs(DATA_PATH_SAVE_EVALUATION)

    print('---------------------------------------------------------------------')
    print('Loading data...')
    print('---------------------------------------------------------------------')    

    imgs_les = load_data(data_dir=DATA_PATH_LES)
    print('LES data loaded!')
    imgs_gan = load_data(data_dir=DATA_PATH_GAN)
    print('GAN data loaded!')
    imgs_vae = load_data(data_dir=DATA_PATH_VAE)
    print('VAE data loaded!')
    imgs_ddpm = load_data(data_dir=DATA_PATH_DDPM)
    print('DDPM data loaded!')

    number_test_images_les = imgs_les.shape[0]
    number_test_images_gan = imgs_gan.shape[0]
    number_test_images_vae = imgs_vae.shape[0]
    number_test_images_ddpm = imgs_ddpm.shape[0]
    
    for fluid_range in fluid_ranges:

        name_mean_v = 'mean_v_' + str(fluid_range)
        name_var_v = 'var_v_python' + str(fluid_range)

        print('---------------------------------------------------------------------')
        print('Mean V')
        print('---------------------------------------------------------------------')
        # Compute mean v for the images
        print('Starting computations for LES images...')
        mean_v_les = computation_mean_v(img_array=imgs_les, x_range=fluid_range)
        print('Starting computations for GAN images...')
        mean_v_gan = computation_mean_v(img_array=imgs_gan, x_range=fluid_range)
        print('Starting computations for VAE images...')
        mean_v_vae = computation_mean_v(img_array=imgs_vae, x_range=fluid_range)
        print('Starting computations for DDPM images...')
        mean_v_ddpm = computation_mean_v(img_array=imgs_ddpm, x_range=fluid_range)

        std_v_les = computation_std_dev_v(img_array=imgs_les, x_range=fluid_range)
        std_v_gan = computation_std_dev_v(img_array=imgs_gan, x_range=fluid_range)
        std_v_vae = computation_std_dev_v(img_array=imgs_vae, x_range=fluid_range)
        std_v_ddpm = computation_std_dev_v(img_array=imgs_ddpm, x_range=fluid_range)

        upper_bound_les, lower_bound_les = compute_confidence_interval(mean_v_les, std_v_les, number_test_images_les, alpha=0.05, mean=True)
        upper_bound_gan, lower_bound_gan = compute_confidence_interval(mean_v_gan, std_v_gan, number_test_images_gan, alpha=0.05, mean=True)
        upper_bound_vae, lower_bound_vae = compute_confidence_interval(mean_v_vae, std_v_vae, number_test_images_vae, alpha=0.05, mean=True)
        upper_bound_ddpm, lower_bound_ddpm = compute_confidence_interval(mean_v_ddpm, std_v_ddpm, number_test_images_ddpm, alpha=0.05, mean=True)

        mean_v_list = [mean_v_les, mean_v_gan, mean_v_vae, mean_v_ddpm]
        upper_bound_list = [upper_bound_les, upper_bound_gan, upper_bound_vae, upper_bound_ddpm]
        lower_bound_list = [lower_bound_les, lower_bound_gan, lower_bound_vae, lower_bound_ddpm]

        # Create the plot
        print('Creating the plot...')
        plot_metric_v(v_val=mean_v_list, save_dir=DATA_PATH_SAVE_EVALUATION, name=name_mean_v)

        print('---------------------------------------------------------------------')
        print('Var V')
        print('---------------------------------------------------------------------')
        # Compute variance v for the LES images
        print('Starting computations for LES images...')
        var_v_les = computation_var_v(img_array=imgs_les, x_range=fluid_range)
        print('Starting computations for GAN images...')
        var_v_gan = computation_var_v(img_array=imgs_gan, x_range=fluid_range)
        print('Starting computations for VAE images...')
        var_v_vae = computation_var_v(img_array=imgs_vae, x_range=fluid_range)
        print('Starting computations for DDPM images...')
        var_v_ddpm = computation_var_v(img_array=imgs_ddpm, x_range=fluid_range)

        upper_bound_les, lower_bound_les = compute_confidence_interval(var_v_les, std_v_les, number_test_images_les, alpha=0.05, mean=False)
        upper_bound_gan, lower_bound_gan = compute_confidence_interval(var_v_gan, std_v_gan, number_test_images_gan, alpha=0.05, mean=False)
        upper_bound_vae, lower_bound_vae = compute_confidence_interval(var_v_vae, std_v_vae, number_test_images_gan, alpha=0.05, mean=False)
        upper_bound_ddpm, lower_bound_ddpm = compute_confidence_interval(var_v_ddpm, std_v_ddpm, number_test_images_gan, alpha=0.05, mean=False)

        var_v_list = [var_v_les, var_v_gan, var_v_vae, var_v_ddpm]
        upper_bound_list = [upper_bound_les, upper_bound_gan, upper_bound_vae, upper_bound_ddpm]
        lower_bound_list = [lower_bound_les, lower_bound_gan, lower_bound_vae, lower_bound_ddpm]

        # Create the plot
        print('Creating the plot...')
        plot_metric_v(v_val=var_v_list, save_dir=DATA_PATH_SAVE_EVALUATION, name=name_var_v)


print('---------------------------------------------------------------------')
print('Done!')
print('---------------------------------------------------------------------')   