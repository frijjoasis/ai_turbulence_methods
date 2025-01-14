# Final version for evaluations published in paper 'Generative Modeling of Turbulence' (Drygala, Winhart, Di Mare, Gottschalk)
# Created by Claudia Drygala

from cv2 import rotate
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import scipy.stats

# Set the environment
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
                cv2.imwrite('file.png', img_revert)
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


def plot_metric_v(v_les, v_gan, save_dir, upper_bound_les, lower_bound_les, upper_bound_gan, lower_bound_gan, name='mean_v'):

    img_height = v_les.shape[0]   # Parameter for normalizing y axis, delta_v_les.shape = delta_v_gan.shape

    y_delta = np.arange(start=0, stop=img_height, step=1)/(img_height-1)

    plt.plot(y_delta, v_les, label='LES', color='r')
    plt.plot(y_delta, v_gan, label='GAN', color='b', linestyle='dashed')
    plt.fill_between(y_delta, lower_bound_gan, upper_bound_gan, color='b', alpha=.1)
    plt.fill_between(y_delta, lower_bound_les, upper_bound_les, color='r', alpha=.29)
    plt.legend()
    plt.xlabel(r'$y/\delta$')
    plt.ylabel(r'$\psi$')
    plt.savefig(os.path.join(save_dir, name + '.jpeg'))
    plt.close()


if __name__ == "__main__":

    # Investigated range of fluid (equals x_range of image)
    fluid_ranges = [300, 500]

    # Set the data paths
    DATA_PATH_LES = '/home/.datasets/Turbulence_AI/Turbulence_AI_original/Turbulence_AI_subset_5000/subset'
    DATA_PATH_GAN = os.path.join('/home/drygala/claudia_data/test') 
    DATA_PATH_SAVE_EVALUATION = './evaluation_karman'
    if not os.path.exists(DATA_PATH_SAVE_EVALUATION):
        os.makedirs(DATA_PATH_SAVE_EVALUATION)

    print('---------------------------------------------------------------------')
    print('Loading data...')
    print('---------------------------------------------------------------------')    

    imgs_les = load_data(data_dir=DATA_PATH_LES)
    imgs_gan = load_data(data_dir=DATA_PATH_GAN)

    number_test_images_les = imgs_les.shape[0]
    number_test_images_gan = imgs_gan.shape[0]

    for fluid_range in fluid_ranges:

        name_mean_v = 'mean_v_' + str(fluid_range)
        name_var_v = 'var_v_python' + str(fluid_range)

        print('---------------------------------------------------------------------')
        print('Mean V')
        print('---------------------------------------------------------------------')
        # Compute mean v for the LES images
        print('Starting computations for LES images...')
        mean_v_les = computation_mean_v(img_array=imgs_les, x_range=fluid_range)

        # Compute mean v for the GAN images
        print('Starting computations for GAN images...')
        mean_v_gan = computation_mean_v(img_array=imgs_gan, x_range=fluid_range)

        std_v_les = computation_std_dev_v(img_array=imgs_les, x_range=fluid_range)
        std_v_gan = computation_std_dev_v(img_array=imgs_gan, x_range=fluid_range)

        upper_bound_les, lower_bound_les = compute_confidence_interval(mean_v_les, std_v_les, number_test_images_les, alpha=0.05, mean=True)
        upper_bound_gan, lower_bound_gan = compute_confidence_interval(mean_v_gan, std_v_gan, number_test_images_gan, alpha=0.05, mean=True)

        # Create the plot
        print('Creating the plot...')
        plot_metric_v(v_les=mean_v_les, v_gan=mean_v_gan, save_dir=DATA_PATH_SAVE_EVALUATION, upper_bound_les=upper_bound_les, lower_bound_les=lower_bound_les, upper_bound_gan=upper_bound_gan, lower_bound_gan=lower_bound_gan, name=name_mean_v)

        print('---------------------------------------------------------------------')
        print('Var V')
        print('---------------------------------------------------------------------')
        # Compute mean v for the LES images
        print('Starting computations for LES images...')
        var_v_les = computation_var_v(img_array=imgs_les, x_range=fluid_range)

        # Compute mean v for the GAN images
        print('Starting computations for GAN images...')
        var_v_gan = computation_var_v(img_array=imgs_gan, x_range=fluid_range)

        upper_bound_les, lower_bound_les = compute_confidence_interval(var_v_les, std_v_les, number_test_images_les, alpha=0.05, mean=False)
        upper_bound_gan, lower_bound_gan = compute_confidence_interval(var_v_gan, std_v_gan, number_test_images_gan, alpha=0.05, mean=False)

        # Create the plot
        print('Creating the plot...')
        plot_metric_v(v_les=var_v_les, v_gan=var_v_gan, save_dir=DATA_PATH_SAVE_EVALUATION, upper_bound_les=upper_bound_les, lower_bound_les=lower_bound_les, upper_bound_gan=upper_bound_gan, lower_bound_gan=lower_bound_gan, name=name_var_v)


print('---------------------------------------------------------------------')
print('Done!')
print('---------------------------------------------------------------------')   