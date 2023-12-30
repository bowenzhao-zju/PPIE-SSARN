from ntpath import join
import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
import time
import options, utils
from criteria import L1_Charbonnier_loss
from dataset.arad_dataset import ARAD_DATASET
from NTIRE2022Util import compute_psnr, compute_sam
from NTIRE2022Util import saveCube

from math import log10
from thop import profile

def main() -> object:
    global opt, model, logger
    ## ====== init setting =======
    
    opt_class = options.BaseOptions()
    opt = opt_class.parse()
    opt_class.print_options(opt)
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    logger = utils.get_logger("test_logger", os.path.join(opt.log_dir, "test.log"))
    cudnn.benchmark = True
    logger.info("===> Testing")
    logger.info("===> Loading datasets")
    logger.info("===> Setting GPU")
    
    torch.cuda.set_device(opt.cuda_id)
    
    test_set = ARAD_DATASET(opt.test_path, mode='test', augment=False)
    test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, pin_memory=True)

    logger.info("===> Loading best model")
    best_model_filename = join(opt.best_model_dir, 'valid_best.pth')
    checkpoint = torch.load(best_model_filename, map_location='cuda:'+str(opt.cuda_id))
    model = checkpoint["model"]
    logger.info("===> Best model (epoch {}) loaded".format(checkpoint["epoch"]))
    test(test_set.image_filenames, test_data_loader, model)

def test(save_path_list, testing_data_loader, model):
    test_bar = tqdm(testing_data_loader)
    test_results = {'filename':[], 'psnr':[], 'sam':[], 'time':[]}
    save_root = join(opt.best_model_dir, 'pred/')
    save_root_png = join(save_root, 'png/')

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(save_root_png):
        os.makedirs(save_root_png)

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_bar):
            input_raw, input, label = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
            N, C, H, W = batch[0].size()
            scale_coord_map = input_matrix_wpn(H, W)

            if opt.cuda:
                input_raw = input_raw.cuda()
                input = input.cuda()
                scale_coord_map = scale_coord_map.cuda()

            last_time = time.time()
            _, pred = model(input_raw, input, scale_coord_map)
            current_time = time.time()
            flops, params = profile(model, inputs=(input_raw, input, scale_coord_map, ))
            # print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) #flops G，para M

            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            pred = np.squeeze(pred)
            label = np.squeeze(label)

            pred = pred[:,10:-10,10:-10]
            label = label[:,10:-10,10:-10]

            test_results['filename'].append(save_path_list[i])
            test_results["psnr"].append(compute_psnr(pred, label, 1))
            test_results["sam"].append(compute_sam(pred, label))
            test_results["time"].append(current_time - last_time)

            pred = np.transpose(pred, (1,2,0))

            save_path_cube = join(save_root, save_path_list[i]+'_16.mat')
            saveCube(save_path_cube, pred)

            for c in range(16):
                save_path_pred = join(save_root_png, save_path_list[i]+str(c).zfill(2)+'.png')
                pred_png = pred[:, :, c]*255
                cv2.imwrite(save_path_pred, pred_png.clip(0, 255).astype(np.uint8))

    test_results['filename'].append(['average'])
    test_results["psnr"].append(np.mean(test_results["psnr"]))
    test_results["sam"].append(np.mean(test_results["sam"]))
    test_results["time"].append(np.mean(test_results["time"]))
    pd.DataFrame(test_results).to_csv(join(save_root, 'ours'+'_result.csv'))

def input_matrix_wpn(inH, inW, add_id_channel=False):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    h_offset_coord[0::4, :, 0] = 0.25
    h_offset_coord[1::4, :, 0] = 0.5
    h_offset_coord[2::4, :, 0] = 0.75
    h_offset_coord[3::4, :, 0] = 1.0

    w_offset_coord[:, 0::4, 0] = 0.25
    w_offset_coord[:, 1::4, 0] = 0.5
    w_offset_coord[:, 2::4, 0] = 0.75
    w_offset_coord[:, 3::4, 0] = 1.0

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)

    return pos_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

if __name__ == "__main__":
    main()
