import torch
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
import logging

def get_logger(logger_name, filename=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def msfaTOcube(raw, msfa_size):
    mask = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=np.int)
    cube = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=np.int)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i * msfa_size + j] = 1
    for i in range(msfa_size**2):
        cube[:, :, i] = raw * (mask[:, :, i])
    return cube

def mask_input(mosaic_image, msfa_size):
    mask = np.zeros((mosaic_image.shape[0], mosaic_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0,msfa_size):
        for j in range(0,msfa_size):
            mask[i::msfa_size, j::msfa_size, i*msfa_size+j] = 1
    input_image = mask * mosaic_image
    return input_image

def reorder_imec(old):
    ### reorder the multiband cube, making the center wavelength from small to large
    '''
    C9  C11  C13  C15
    C8  C10  C12  C14
    C1  C3   C5   C7
    C0  C2   C4   C6
    '''
    _,_,C = old.shape
    assert C==16
    new = np.zeros_like(old)
    new[:, :, 0] = old[:, :, 12]
    new[:, :, 1] = old[:, :, 8]
    new[:, :, 2] = old[:, :, 13]
    new[:, :, 3] = old[:, :, 9]
    new[:, :, 4] = old[:, :, 14]
    new[:, :, 5] = old[:, :, 10]
    new[:, :, 6] = old[:, :, 15]
    new[:, :, 7] = old[:, :, 11]
    new[:, :, 8] = old[:, :, 4]
    new[:, :, 9] = old[:, :, 0]
    new[:, :, 10] = old[:, :, 5]
    new[:, :, 11] = old[:, :, 1]
    new[:, :, 12] = old[:, :, 6]
    new[:, :, 13] = old[:, :, 2]
    new[:, :, 14] = old[:, :, 7]
    new[:, :, 15] = old[:, :, 3]
    return new