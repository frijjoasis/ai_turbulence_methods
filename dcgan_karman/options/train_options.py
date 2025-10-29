"""
The options we can set for the training. The options can be directly changed by appending the desired settings in the functional call
in the terminal in the type of --gpu_id...

The settings of an experiment will be saved in ../train_output/options_log

Implementation by Claudia Drygala, University of Wuppertal
"""


import argparse
import os

class TrainOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to run the training on')
        parser.add_argument('--experiment_name', type=str, default='dcgan', help='name of the experiment under which the results will be saved')
        parser.add_argument('--dataset_dir', required=True, type=str, default="\home\.datasets", help='the directory to the dataset \path_to\data_folder. Datasets should have the following structure: data_folder->data_subfolder->images')
        parser.add_argument('--outer_key', type=int, default=0, help='Choose which dataset to use, when using the ilmenau dataset, possible values: 0-8')
        parser.add_argument('--resize_img', type=int, default=512, help='size k to which the input image should be resized. DCGAN is trained on k x k images')
        parser.add_argument('--batch_size', type=int, default=20, help='the batch size used during training')
        parser.add_argument('--lr', type=int, default=0.0002, help='the learning rate')
        parser.add_argument('--beta1', type=int, default=0.5, help='parameter beta1 of adam optimizer')
        parser.add_argument('--beta2', type=int, default=0.999, help='parameter beta2 of adam optimizer')
        parser.add_argument('--channels', type=int, default=1, help='number of channels of the input image')
        parser.add_argument('--latent_dim', type=int, default=100, help='length of the latent vector which is input for the generator')
        parser.add_argument('--pretrained', type=str, default=False, help='true, if a pretrained model should be used')
        parser.add_argument('--continue_train', type=str, default=False, help='true, if a model should be continued to train')
        parser.add_argument('--epoch_start', type=int, default=200 + 1, help='the epoch of which the last model has been derived+1')
        parser.add_argument('--checkpoint_dir', type=str, help='directory of the checkpoints of the generator and discriminator')
        parser.add_argument('--model_G', type=str, help='name of checkpoint of the generators model')
        parser.add_argument('--model_D', type=str, help='name of checkpoint of the discriminators model')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs the model should train')
        parser.add_argument('--num_test_samples', type=int, default=1, help='number of noise vectors on which the generator is applied during the training to consider the training process')
        parser.add_argument('--tensorboard_active', type=str, default=False, help='if true tensorboard will be activated during training to log data additionally')
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
        expr_dir = '../train_output/options_log/'
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, opt.experiment_name) + '-train_options.txt'
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




