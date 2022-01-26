# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 

'''
from PIL import Image
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.cuda.comm
import torch
from math import exp
import matplotlib
import sys
import torch.nn as nn
matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)


class PSNRMetric(nn.Module):
    
    def __init__(self, max_intensity=1.0):
        super(PSNRMetric, self).__init__()
        self.max_intensity = max_intensity

    @staticmethod
    def compute_mse(original_batch, result_batch, mask):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        original_batch, result_batch = original_batch * mask, result_batch * mask
        # Multiply number of unmasked pixels by number of channels
        unmasked_pixels = original_batch.shape[1] * torch.squeeze(mask, dim=1).sum(dim=(1,2))
        return ((original_batch - result_batch) ** 2).sum(dim=(1,2,3))/unmasked_pixels

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, mask_batch, max_intensity=1.0):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """        
        image_batchA, image_batchB = torch.clamp(image_batchA, 0.0, 1.0), \
                                     torch.clamp(image_batchB, 0.0, 1.0)
        # Calculate PSNR per image
        psnr_val = 10 * torch.log10(max_intensity ** 2 /
                                    PSNRMetric.compute_mse(image_batchA, image_batchB, mask_batch))

        # Take average over batch dimension, ignoring NaNs
        psnr_mean = psnr_val.nanmean()
        return psnr_mean if not psnr_mean.isnan() else None
    
    def forward(self, image_batchA, image_batchB, mask_batch):
        return PSNRMetric.compute_psnr(image_batchA, image_batchB, 
                                       mask_batch, max_intensity=self.max_intensity)


class MSSSIMMetric(nn.Module):
    def __init__(self, window_size=11, num_channel=3):
        super(MSSSIMMetric, self).__init__()
        self.msssim_weights = nn.Parameter(torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]), 
                                           requires_grad=False)
        self.levels = self.msssim_weights.size()[0]
        
        # Default values for ssim methods
        self.window_size = window_size
        self.num_channel = num_channel
        self.gaussian_window = MSSSIMMetric.create_window(self.window_size, self.num_channel)

    @staticmethod
    def create_window(window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = MSSSIMMetric.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def gaussian(window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        Note ssim does not assume a color space (https://arxiv.org/pdf/2006.13846.pdf)

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        window = self.gaussian_window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.window_size // 2, groups=self.num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.window_size // 2, groups=self.num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.window_size // 2, groups=self.num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.window_size // 2, groups=self.num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.window_size // 2, groups=self.num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.cuda() + mu2_sq.cuda() + C1) *
                     (sigma1_sq.cuda() + sigma2_sq.cuda() + C2))
        ssim_map = ssim_map1.cuda() / ssim_map2.cuda()

        v1 = 2.0 * sigma12.cuda() + C2
        v2 = sigma1_sq.cuda() + sigma2_sq.cuda() + C2
        cs = torch.mean(v1 / v2, dim=(1, 2, 3))

        return ssim_map.mean(dim=(1, 2, 3)), cs

    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.shape[2]!=img2.shape[2]:
                img1=img1.transpose(2,3)

        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

        device = img1.device
        ssims = []
        mcs = []
        for _ in range(self.levels):
            ssim, cs = self.compute_ssim(img1, img2)

            # Relu normalize (not compliant with original definition)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims, dim=1)
        mcs = torch.stack(mcs, dim=1)

        # Simple normalize (not compliant with original definition)
        # TODO: remove support for normalize == True (kept for backward support)
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** self.msssim_weights.reshape(1, self.msssim_weights.shape[0])
        pow2 = ssims ** self.msssim_weights.reshape(1, self.msssim_weights.shape[0])

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:, :-1] * pow2[:, -1].reshape(-1, 1), dim=1)
        return output
    
    def forward(self, img1, img2):
        return self.compute_msssim(img1, img2)

