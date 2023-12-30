import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from model.ddf import DDFPack
import cv2

def get_WB_filter(size):
    """make a 2D weight bilinear kernel suitable for interpolation"""
    ligne = []
    colonne = []
    half = (size+1) // 2
    for i in range(size):
        if (i + 1) <= half:
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (half**2))
    filter_WB = np.reshape(BilinearFilter, (size, size))
    filter_WB = torch.from_numpy(filter_WB).float()
    filter_WB = filter_WB.view(1, 1, size, size).repeat(16, 1, 1, 1)
    return filter_WB


def get_PPI_filter(size):
    """make a 2D weight kernel suitable for PPI estimation"""
    ligne = [1/2, 1, 1, 1, 1/2]
    colonne = [1/2, 1, 1, 1, 1/2]
    PPIFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            PPIFilter[(j + i * size)] = (ligne[i] * colonne[j] / 16)
    filter_PPI = np.reshape(PPIFilter, (size, size))
    filter_PPI = torch.from_numpy(filter_PPI).float()
    filter_PPI = filter_PPI.view(1, 1, 5, 5)
    return filter_PPI


def get_edge_filter(mode):
    """make a 2D weight kernel suitable for edge extraction"""
    if mode == 'x':
        filter_edge = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif mode == 'y':
        filter_edge = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter_edge = torch.from_numpy(filter_edge).float()
    filter_edge = filter_edge.view(1, 1, 3, 3).repeat(16, 1, 1, 1)
    return filter_edge


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class PPIGenerate(nn.Module):
    def __init__(self, num_features):
        super(PPIGenerate, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=5, padding=2),
        )
        self.ppig_5x5 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False, groups=1)
        self.ppig_5x5.weight.data = get_PPI_filter(5)

    def forward(self, x):
        residual = self.ppig_5x5(x)
        return residual + self.module(x)


class GradientMapGenerate(nn.Module):
    def __init__(self):
        super(GradientMapGenerate, self).__init__()
        self.g_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.g_x.weight.data = get_edge_filter(mode='x')
        self.g_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.g_y.weight.data = get_edge_filter(mode='y')

    def forward(self, x):
        return torch.abs(self.g_x(x)) + torch.abs(self.g_y(x))

# The model name was adjusted during the revision process of the paper.
class RCAN_MS(nn.Module):
    def __init__(self):
        super(RCAN_MS, self).__init__()
        num_features = 64 # number of feature maps
        num_rg = 2 # number of residual groups
        num_rcab = 5 # number of residual blocks
        reduction = 16 # number of feature maps reduction
        
        self.pg = PPIGenerate(16)
        self.gmg = GradientMapGenerate()

        self.interp_7x7 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3, bias=False, groups=16)
        self.interp_7x7.weight.data = get_WB_filter(7)

        self.conv0 = nn.Conv2d(1, 16, kernel_size=1, padding=0)
        self.ddf = DDFPack(16, kernel_size=5, stride=1, kernel_combine='add')

        self.sf = nn.Conv2d(16, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, 16, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

    def forward(self, x, y, pos_mat): # x->input_raw y->input_hard_split pos_mat->only be used in MCAN
        # PPI Edge Infusion Subbranch
        ppi = self.pg(x)
        p = ppi.detach()
        high_freq = self.gmg(p)

        # Spatial–Spectral Adaptive Residual Subbranch

        # Hard Splitting and Interpolation
        low_freq = self.interp_7x7(y)
                
        # Decoupled Dynamic Mosaic Filter
        x = self.conv0(x)
        x = self.ddf(x)
        
        # Residual Channel Attention Network
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.conv2(x)
        mid = x.detach()
        x += low_freq
        x += high_freq
        x = self.conv3(x)

        return ppi, x