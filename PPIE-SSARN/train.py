import os
import random
import json
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
from model.RCAN_MS import RCAN_MS
from model.MCAN import MCAN
from criteria import L1_Charbonnier_loss
from dataset.arad_dataset import ARAD_DATASET
from NTIRE2022Util import compute_psnr

def main() -> object:
    global opt, model, logger
    ## ====== init setting =======
    opt_class = options.BaseOptions()
    opt = opt_class.parse()
    opt_class.print_options(opt)
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    writer = SummaryWriter(opt.log_dir)
    logger = utils.get_logger("train_logger", os.path.join(opt.log_dir, "train.log"))

    opt.seed = random.randint(1, 10000)
    logger.info("Random seed: {%d}" % (opt.seed))
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    logger.info("===> Loading datasets")
    opt.augment_flag = True
    train_set = ARAD_DATASET(opt.train_path, mode="train", augment=opt.augment_flag)
    valid_set = ARAD_DATASET(opt.valid_path, mode="test")

    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=14, pin_memory=True)
    valid_data_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, pin_memory=True)

    logger.info("===> Building model")
    model = choose_model(opt.model_name)
    criterion = choose_loss(opt.loss_name)
    logger.info("===> Printing model\n{%s}" % (model))
    logger.info("===> Setting GPU")
    if opt.cuda:
        device_flag = torch.device('cuda')
        torch.cuda.set_device(opt.cuda_id)
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        device_flag = torch.device('cpu')
        model = model.cpu()
    
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.break_model_dir):
            logger.info("===> Loading checkpoint '{}'".format(opt.break_model_dir))
            checkpoint = torch.load(opt.break_model_dir, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            logger.info("===> No checkpoint found at '{}'".format(opt.break_model_dir))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            logger.info("===> Loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained, map_location=lambda storage, loc: storage)
            model.load_state_dict(weights['model'].state_dict())
        else:
            logger.info("===> No model found at '{}'".format(opt.pretrained))

    logger.info("===> Setting optimizer")
    for name, param in model.named_parameters():
        if "interp_7x7" in name:
            param.requires_grad = False
        elif "ppig_5x5" in name:
            param.requires_grad = False
        elif "g_x" in name:
            param.requires_grad = False
        elif "g_y" in name:
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    save_opt(opt)
    logger.info("===> Training")
    logger.info('# parameters:{%d}' % (sum(param.numel() for param in model.parameters()))) 
    logger.info("epoch_iters: {}".format(len(training_data_loader)))
    results = {'loss1': [], 'loss2': [], 'all_loss': [], 'psnr_train_ppi': [], 'psnr_train': [], 'psnr_valid': []}
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        logger.info("============================================================================")
        running_results = train(training_data_loader, optimizer, model, criterion, epoch, opt.nEpochs)
        results['loss1'].append(running_results['loss1'] / running_results['batch_sizes'])
        results['loss2'].append(running_results['loss2'] / running_results['batch_sizes'])
        results['psnr_train_ppi'].append(running_results['all_psnr_ppi'] / running_results['batch_sizes'] * opt.batchSize)
        results['psnr_train'].append(running_results['all_psnr'] / running_results['batch_sizes'] * opt.batchSize)
        results['all_loss'].append(running_results['all_loss'] / running_results['batch_sizes'])
        valid_results = valid(valid_data_loader, optimizer, model, criterion, epoch, opt.nEpochs)
        results['psnr_valid'].append(valid_results['average_psnr'])
        
        # tensorboard writter
        writer.add_scalar("train/loss1",  running_results['loss1'] / running_results['batch_sizes'], epoch)
        writer.add_scalar("train/loss2",  running_results['loss2'] / running_results['batch_sizes'], epoch)
        writer.add_scalar("train/loss",  running_results['all_loss'] / running_results['batch_sizes'], epoch)
        writer.add_scalar("train/psnr_ppi",  running_results['all_psnr_ppi'] / running_results['batch_sizes'] * opt.batchSize, epoch)
        writer.add_scalar("train/psnr",  running_results['all_psnr'] / running_results['batch_sizes'] * opt.batchSize, epoch)
        writer.add_scalar("valid/psnr",  valid_results['average_psnr'], epoch)
        logger.info("Epoch:%d Train loss:%.4f Valid psnr:%.4f" %
            (epoch, running_results['all_loss'] / running_results['batch_sizes'], valid_results['average_psnr']))
        
        save_last_checkpoint(model, epoch, valid_results['average_psnr'])
        
        if np.argmax(results['psnr_valid']) == len(results['psnr_valid'])-1:
            save_checkpoint(model, epoch, max(results['psnr_valid']))
    writer.close()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch, num_epochs):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    train_bar = tqdm(training_data_loader)
    running_results = {'batch_sizes': 0, 'loss1': 0, 'loss2': 0, 'all_loss': 0, 'psnr_ppi': 0, 'all_psnr_ppi': 0, 'psnr': 0, 'all_psnr': 0}
    model.train()
    period_time = time.time()

    for i, batch in enumerate(train_bar):
        input_raw, input, label = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
        N, C, H, W = batch[0].size()
        running_results['batch_sizes'] += N

        scale_coord_map = input_matrix_wpn(H, W)
        if opt.cuda:
            input_raw = input_raw.cuda()
            input = input.cuda()
            label = label.cuda()
            label_ppi = label.clone().mean(1).unsqueeze(1)
            scale_coord_map = scale_coord_map.cuda()

        optimizer.zero_grad()
        pred_ppi, pred = model(input_raw, input, scale_coord_map)
        loss1 = 0.125 * criterion(pred_ppi, label_ppi)
        loss2 = 0.125 * criterion(pred, label)
        loss = loss1 + loss2
        
        loss.backward()
        optimizer.step()

        running_results['loss1'] += loss1.item()
        running_results['loss2'] += loss2.item()
        running_results['all_loss'] += loss.item()
        
        pred_ppi = pred_ppi.detach().cpu().numpy()
        label_ppi = label_ppi.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        running_results['psnr_ppi'] = compute_psnr(pred_ppi, label_ppi, 1)
        running_results['all_psnr_ppi'] += running_results['psnr_ppi']
        running_results['psnr'] = compute_psnr(pred, label, 1)
        running_results['all_psnr'] += running_results['psnr']

        train_bar.set_description(desc='[%d/%d] Loss: %.4f PSNR_PPI: %.4f PSNR: %.4f' % (
            epoch, num_epochs, 
            running_results['all_loss'] / running_results['batch_sizes'],
            running_results['all_psnr_ppi'] / running_results['batch_sizes'] * N,
            running_results['all_psnr'] / running_results['batch_sizes'] * N))

        if i % opt.print_freq == 0:
            epoch_iters = len(training_data_loader)
            progress = (i + 1) / epoch_iters * 100
            logger.info("Train epoch: %d %d/%d %d%%  Period loss1:%.4f  Period loss2:%.4f  Period loss:%.4f  Period PSNR:%.4f  Period time:%.4f" % 
                        (epoch, i+1, epoch_iters, progress,
                        running_results['loss1'] / running_results['batch_sizes'], 
                        running_results['loss2'] / running_results['batch_sizes'], 
                        running_results['all_loss'] / running_results['batch_sizes'], 
                        running_results['all_psnr'] / running_results['batch_sizes'] * N,
                        time.time()-period_time))
            period_time = time.time()

    return running_results

def valid(testing_data_loader, optimizer, model, criterion, epoch, num_epochs):
    test_bar = tqdm(testing_data_loader)
    test_results = {'batch_sizes': 0, 'psnr': 0, 'all_psnr': 0, 'average_psnr': 0}
    model.eval()

    with torch.no_grad():
        for batch in test_bar:
            input_raw, input, label = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
            N, C, H, W = batch[0].size()
            test_results['batch_sizes'] += N
            scale_coord_map = input_matrix_wpn(H, W)

            if opt.cuda:
                input_raw = input_raw.cuda()
                input = input.cuda()
                label = label.cuda()
                scale_coord_map = input_matrix_wpn(H, W)

            _, pred = model(input_raw, input, scale_coord_map)

            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            test_results['psnr'] = compute_psnr(pred, label, 1)
            test_results['all_psnr'] += test_results['psnr']
            test_results['average_psnr'] = test_results['all_psnr'] / test_results['batch_sizes'] * N

            test_bar.set_description(desc='[%d/%d] PSNR: %.4f' % (
                epoch, num_epochs, test_results['psnr']))

    return test_results

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

def save_checkpoint(model, epoch, psnr):
    model_out_path = opt.model_dir + "/valid_best.pth"
    state = {"epoch": epoch ,"model": model}
    best_txt_path = opt.model_dir + '/valid_best.txt'
    with open(best_txt_path, 'w') as txtfile:
        txtfile.write("epoch: {}\nPSNR: {}\n".format(epoch, psnr))
    torch.save(state, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def save_last_checkpoint(model, epoch, psnr):
    model_out_path = opt.model_dir + "/last.pth"
    state = {"epoch": epoch ,"model": model}
    last_txt_path = opt.model_dir + '/last.txt'
    with open(last_txt_path, 'w') as txtfile:
        txtfile.write("epoch: {}\nPSNR: {}\n".format(epoch, psnr))
    torch.save(state, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def save_opt(opt):
    with open(opt.log_dir + '/opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    print("save opt")

def save_statistics(opt, results, epoch):
    data_frame = pd.DataFrame(
        data=results,index=range(opt.start_epoch, epoch + 1))
    data_frame.to_csv(opt.log_dir + '/train_results.csv', index_label='Epoch')

def choose_model(model_name):
    if model_name == "MCAN":
        model = MCAN()
    elif model_name == "PPIE_SSARN":
        model = RCAN_MS()
    return model

def choose_loss(loss_name):
    if loss_name == "L1_Charbonnier_loss":
        criterion = L1_Charbonnier_loss()
    return criterion

if __name__ == "__main__":
    main()
