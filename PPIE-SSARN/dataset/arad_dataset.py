from os import listdir
from os.path import join
import torch

import torch.utils.data as data
import hdf5storage
import numpy as np
import random

from utils import mask_input

import pandas as pd
import sys
import config

sys.path.append('../')

def load_raw(filepath):
    mat = hdf5storage.loadmat(filepath)
    img = mat['mosaic']
    return img

def load_target(filepath):
    mat = hdf5storage.loadmat(filepath)
    # ARAD Dataset
    img = mat['cube']
    norm_factor = mat['norm_factor']

    # Chikusei Dataset
    # data = mat['crop_gt']
    # img = data[0,0]['cube']
    # norm_factor = data[0,0]['norm_factor']
    return img, norm_factor

def rand_crop(target, raw, crop_size):
    [h, w, _] = target.shape
    Height = random.randint(0, (h - crop_size[0]))
    Width = random.randint(0, (w - crop_size[1]))
    return target[Height:(Height + crop_size[0]), Width:(Width + crop_size[1]), :], raw[Height:(Height + crop_size[0]), Width:(Width + crop_size[1])]

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

class ARAD_DATASET(data.Dataset):
    def __init__(self, image_dir, mode='train', augment=False):
        super(ARAD_DATASET, self).__init__()
        if mode == 'train':
            mosaic_dir = join(image_dir, "train_mosaic")
            target_dir = join(image_dir, "train_spectral_16")
        elif mode == 'test':
            mosaic_dir = join(image_dir, "test_mosaic")
            target_dir = join(image_dir, "test_spectral_16")    
        self.image_filenames = [x.split('.')[0] for x in listdir(mosaic_dir)]
        self.mosaic_files = [join(mosaic_dir, fn+".raw.mat") for fn in self.image_filenames]
        self.target_files = [join(target_dir, fn+"_16.mat") for fn in self.image_filenames]
        self.crop_size = config.CROP_SIZE
        self.augment = augment

    def __getitem__(self, index):
        raw = load_raw(self.mosaic_files[index])
        raw = raw.astype(np.float32)
        target, _ = load_target(self.target_files[index])
        target = target.astype(np.float32)
        
        # Target Augmentation
        if self.augment:
            if np.random.uniform() < 0.5:
                target = np.fliplr(target)
                raw = np.fliplr(raw)
            if np.random.uniform() < 0.5:
                target = np.flipud(target)
                raw = np.flipud(raw)
            k = random.randint(1, 4)
            target = np.rot90(target, k)
            raw = np.rot90(raw, k)
            target, raw = rand_crop(target, raw, self.crop_size)

        # Create RAW Mosaic from Cube
        raw = np.expand_dims(raw, axis=2)
        input_image = mask_input(raw, 4)    # mosaic hard split: [h, w, 1]->[h, w, 16]

        raw = np.transpose(raw, (2, 0, 1)).astype(np.float32)
        input_image = np.transpose(input_image, (2, 0, 1)).astype(np.float32)  
        target = np.transpose(target, (2, 0, 1)).astype(np.float32)
        raw = torch.from_numpy(raw)
        input_image = torch.from_numpy(input_image)
        target = torch.from_numpy(target)

        return raw, input_image, target

    def __len__(self):
        return len(self.mosaic_files)