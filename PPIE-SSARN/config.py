# -*-coding:utf-8-*-
# Dataset:
TRAIN_PATH                  = "/data1/zbw/dataset/demosaic/ARAD/train/"
VALID_PATH                  = "/data1/zbw/dataset/demosaic/ARAD/test/"
TEST_PATH                   = "/data1/zbw/dataset/demosaic/ARAD/test/"
# TRAIN_PATH                  = "/data1/zbw/dataset/demosaic/Chikusei/train/"
# VALID_PATH                  = "/data1/zbw/dataset/demosaic/Chikusei/test/"
# TEST_PATH                   = "/data1/zbw/dataset/demosaic/Chikusei/test/"
CROP_SIZE                   = (128, 128)

# Training:
MODEL_NAME    = "PPIE_SSARN" # MCAN
LOSS_NAME     = "L1_Charbonnier_loss"
CUDA_ID       = 3
BATCH_SIZE    = 32
START_EPOCH   = 1
END_EPOCH     = 10000
LR_DECAY_STEP = 300
PRINT_FREQ    = 5
RESUME        = ""
LOG_ROOT      = "./log/"
MODEL_DIR     = "./checkpoint/"

# Resume:
RESUME          = False
BREAK_MODEL_DIR = "./checkpoint/valid_best.pth"

# Testing:
BEST_MODEL_DIR = "/PPIE-SSARN/checkpoint/"

# Optimizer:
LR            = 1.0e-3
WEIGHT_DECAY  = 1.0e-4
MOMENTUM      = 0.9