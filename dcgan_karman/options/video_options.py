"""
The options we can set for the training. The options can be directly changed by appending the desired settings in the functional call
in the terminal in the type of --gpu_id...

The settings of an experiment will be saved in ../train_output/options_log

Implementation by Claudia Drygala, University of Wuppertal
"""


import argparse
import os

class VideoOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser = argparse.ArgumentParser()
        parser.add_argument('--images_dir', required=True, type=str, help='the directory to the images by which the video should be created')
        parser.add_argument('--video_name', required=True, type=str, help='name of the video under which the results will be saved')
        parser.add_argument('--fps', type=int, default=15, help='frames per second, frame rate of video')
        parser.add_argument('--video_type', type=str, default='avi', help='type of video. Possible choices: avi, mp4, both (for avi and mp4)')
        parser.add_argument('--pix2pixHD', type=bool, default=False, help='set True if the images are outcome of inference from trained pix2pixHD model')
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
        expr_dir = '../videos/videos_log/'
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, opt.video_name) + '_' + opt.video_type +'.txt'
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




