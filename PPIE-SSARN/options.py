import argparse
from json import load
import os
import config
import time


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        ####kpn networks####
        ## path parameters
        # Training settings
        parser = argparse.ArgumentParser(description="PyTorch LapSRN")
        parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
        parser.add_argument("--nEpochs", type=int, default=7500, help="number of epochs to train for")
        parser.add_argument("--lr", type=float, default=2e-3, help="Learning Rate. Default=1e-4")
        parser.add_argument("--step", type=int, default=2000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
        parser.add_argument("--cuda", type=bool, default=True, help="Use cuda?")
        parser.add_argument("--resume", default=False, type=bool, help="Path to checkpoint (default: none)")
        parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
        parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
        parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
        parser.add_argument("--weight_decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
        parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
        parser.add_argument('--msfa_size', '-uf',  type=int, default=4, help="the size of square msfa")
        parser.add_argument("--data_path", default="/data1/NTIRE2022/dataset/train/", type=str, help="path to train dataset")

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # overwrite parameters from config file
        self.load_cfg(opt)

        # save and return the parser
        self.parser = parser
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
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
        return message

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        return self.gather_options()


    def load_cfg(self, opt):
        opt.train_path      = config.TRAIN_PATH
        opt.valid_path      = config.VALID_PATH
        opt.test_path       = config.TEST_PATH
        opt.model_name      = config.MODEL_NAME
        opt.loss_name       = config.LOSS_NAME
        opt.cuda_id         = config.CUDA_ID
        opt.batchSize       = config.BATCH_SIZE
        opt.start_epoch     = config.START_EPOCH
        opt.nEpochs         = config.END_EPOCH
        opt.step            = config.LR_DECAY_STEP
        opt.weight_decay    = config.WEIGHT_DECAY
        opt.lr              = config.LR
        opt.momentum        = config.MOMENTUM
        opt.print_freq      = config.PRINT_FREQ

        opt.resume          = config.RESUME
        opt.break_model_dir = config.BREAK_MODEL_DIR
        opt.best_model_dir  = config.BEST_MODEL_DIR
        
        local_time      = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
        opt.log_dir = os.path.join(config.LOG_ROOT, local_time)
        opt.model_dir = os.path.join(config.MODEL_DIR, local_time)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        





