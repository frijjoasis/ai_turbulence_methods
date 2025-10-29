"""
The options we can set for the inference. The options can be directly changed by appending the desired settings in the functional call
in the terminal in the type of --gpu_id...

The settings of an experiment will be saved in ../inference_output/options_log

Implementation by Claudia Drygala, University of Wuppertal
"""

import argparse
import os

class InferenceOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_dir', type=str, default="\home\.datasets", help='the directory to the dataset \path_to\data_folder. Datasets should have the following structure: data_folder->data_subfolder->images')
        parser.add_argument('--seed', type=str, default=False, help='true if a seed should be set. Makes sense for reproducabiloty of experiments')
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to run the inference on')
        parser.add_argument('--folder_dir', type=str, default="dcgan_original", help='the folder to generated images')
        parser.add_argument('--experiment_name', type=str, default='dcgan', help='name of the experiment under which the results will be saved')
        parser.add_argument('--resize_img', type=int, default=256, help='size k of the images on which the DCGAN has been trained on')
        parser.add_argument('--channels', type=int, default=2, help='number of channels of the input image')
        parser.add_argument('--image_type', type=str, default='karman', help='the type of images the generator has been trained on. Possible choices are: karman, lpt')
        parser.add_argument('--latent_dim', type=int, default=100, help='length of the latent vector which is input for the generator')
        parser.add_argument('--checkpoint_dir', type=str, help='directory of the checkpoints of the generator')
        parser.add_argument('--model_G', type=str, help='name of checkpoint of the generators model')
        parser.add_argument('--num_images', type=int, default=3000, help='number of images to generate')
        parser.add_argument('--outer_key', type=int, default=0, help='Choose which dataset to use, when using the ilmenau dataset, possible values: 0-8')
        parser.add_argument('--latent_type', type=str, default='naive', help='approach to compute the latent vectors. Possible choices are: naive, bilinear, bilinear2P, random_walk')
        parser.add_argument('--bi_steps', type=int, default=10, help='the number of steps between each interpolation of two latent vectors')
        self.initialized = True
        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = '../inference_output/options_log/'
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, opt.experiment_name) + '_' + opt.latent_type + '-inference_options.txt'
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser

        self.print_options(opt)
        self.opt = opt
        return self.opt




