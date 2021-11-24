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
from util import ImageProcessing
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

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
        
    def save_images(output_img_batch, net_output_img_batch, psnrs, ssims, name, epoch):        
        for i in range(0, input_img_batch.shape[0]):

            output_img_example = (output_img_batch[i] * 255).astype('uint8')
            net_output_img_example = (net_output_img_batch[i] * 255).astype('uint8')
            psnr_example = psnrs[i]
            ssim_example = ssims[i]

            save_path = out_dirpath + "/" + name[0].split(".")[0] + "_" + self.split_name.upper() + "_" + str(epoch + 1) + "_PSNR_" \
                        + str("{0:.3f}".format(psnr_example)) + "_SSIM_" + str("{0:.3f}".format(ssim_example)) + ".jpg"
            
            plt.imsave(save_path, ImageProcessing.swapimdims_3HW_HW3(net_output_img_example))

    def evaluate(self, net, epoch=0):
        """Evaluates a network on a specified split of a dataset e.g. test, validation
        :param net: PyTorch neural network data structure
        :param data_loader: an instance of the DataLoader class for the dataset of interest
        :param split_name: name of the split e.g. "test", "validation"
        :param log_dirpath: logging directory
        :returns: average loss, average PSNR
        :rtype: float, float
        """

        out_dirpath = self.log_dirpath + "/" + self.split_name.lower()
        if not os.path.isdir(out_dirpath):
            os.mkdir(out_dirpath)

        # switch model to evaluation mode
        net.eval()
        net.cuda()

        with torch.no_grad():

            batch_pbar = tqdm(enumerate(self.data_loader, 0), total=len(self.data_loader))
            
            for batch_num, data in batch_pbar:

                input_img_batch, output_img_batch, name = Variable(data['input_img'], requires_grad=False).cuda(), \
                                                          Variable(data['output_img'], requires_grad=False).cuda(), \
                                                          data['name']
                        
                input_img_batch = torch.clamp(input_img_batch, 0, 1)
                net_output_img_batch ,_= net(input_img_batch)
                loss = self.criterion(net_output_img_batch, output_img_batch, torch.zeros(net_output_img_batch.shape[0]))
                loss_scalar = loss.item()
                net_output_img_batch = torch.clamp(net_output_img_batch, 0, 1)
                
                psnr_avg = ImageProcessing.compute_psnr(output_img_batch, net_output_img_batch, torch.tensor(1.0)).item()
                ssim_avg = ImageProcessing.compute_ssim(output_img_batch,net_output_img_batch)
                
                batch_pbar.set_description('Train Loss: {}'.format(loss_scalar))

        logging.info('loss_%s: %.5f psnr_%s: %.3f ssim_%s: %.3f' % (
            self.split_name, loss_scalar, self.split_name, psnr_avg, self.split_name, ssim_avg))

        return loss_scalar, psnr_avg, ssim_avg
