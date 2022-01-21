# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

'''
import matplotlib
matplotlib.use('agg')
import numpy as np
import sys
import os
import torch
import logging
from torch.autograd import Variable
import metric
import transpose
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluator():

    def __init__(self, criterion, data_loader, split_name, log_dirpath):
        """Initialisation function for the data loader
        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A
        """
        super().__init__()
        self.criterion = criterion
        self.data_loader = data_loader
        self.split_name = split_name
        self.log_dirpath = log_dirpath
        self.psnr = metric.PSNRMetric().to(device)
        self.msssim = metric.MSSSIMMetric().to(device)
        
    def save_images(self, net_output_img_batch, names, epoch):  
        numpy_batch = net_output_img_batch.cpu().numpy()
        
        split_dirpath = self.log_dirpath + "/" + self.split_name.lower() + "/"
        epoch_dirpath = split_dirpath + "/" + str(epoch + 1) + "/"
        
        for data_dir in [split_dirpath, epoch_dirpath]:
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)

        for i in range(0, numpy_batch.shape[0]):
            net_output_img_example = (numpy_batch[i] * 255).astype('uint8')
            save_path = epoch_dirpath + names[i]
            plt.imsave(save_path, transpose.swapimdims_3HW_HW3(net_output_img_example))

    def evaluate(self, net, epoch=0, save_images=False):
        """Evaluates a network on a specified split of a dataset e.g. test, validation
        :param net: PyTorch neural network data structure
        :param data_loader: an instance of the DataLoader class for the dataset of interest
        :param split_name: name of the split e.g. "test", "validation"
        :param log_dirpath: logging directory
        :returns: average loss, average PSNR, average SSIM
        :rtype: float, float, float
        """

        # switch model to evaluation mode
        net.eval()
        net.to(device)

        with torch.no_grad():

            batch_pbar = tqdm(enumerate(self.data_loader, 0), total=len(self.data_loader))
            examples = 0.0
            running_loss = 0.0
            batches = 0.0
            
            for batch_num, data in batch_pbar:

                input_img_batch, output_img_batch, mask_batch, names = \
                    Variable(data['input_img'], requires_grad=False).to(device, non_blocking=True), \
                    Variable(data['output_img'], requires_grad=False).to(device, non_blocking=True), \
                    Variable(data['mask'], requires_grad=False).to(device, non_blocking=True), \
                    data['name']
                        
                net_output_img_batch = net(input_img_batch, mask_batch)
                net_output_img_batch = torch.clamp(net_output_img_batch, 0.0, 1.0)
                
                loss = self.criterion(net_output_img_batch, output_img_batch, mask_batch)
                loss_scalar = loss.item()
                running_loss += loss_scalar
                examples += input_img_batch.shape[0]
                batches += 1
                                
                psnr_avg = self.psnr(output_img_batch, net_output_img_batch, mask_batch).item()
                msssim_avg = self.msssim(output_img_batch * mask_batch, 
                                         net_output_img_batch * mask_batch).mean().item()
                
                if save_images:
                    self.save_images(net_output_img_batch, names, epoch)
                
                batch_pbar.set_description('Epoch {}. Loss: {}'.format(epoch, loss_scalar))

        logging.info('loss_%s: %.5f psnr_%s: %.3f msssim_%s: %.3f' % (
            self.split_name, running_loss / batches, self.split_name, psnr_avg, self.split_name, msssim_avg))

        return running_loss / batches, psnr_avg, msssim_avg
