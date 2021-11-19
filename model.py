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
import torch
import torch.nn as nn
from collections import defaultdict
import rgb_ted
from util import ImageProcessing
from torch.autograd import Variable
import math
from math import exp
import torch.nn.functional as F

np.set_printoptions(threshold=sys.maxsize)


class CURLLoss(nn.Module):

    def __init__(self, ssim_window_size=5, alpha=0.5):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLoss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

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
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
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

        pow1 = mcs ** weights.reshape(1, weights.shape[0])
        pow2 = ssims ** weights.reshape(1, weights.shape[0])

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:, :-1] * pow2[:, -1].reshape(-1, 1), dim=1)
        return output
    
    @staticmethod
    def batch_lab_convert(img_batch):
        converted_batch = ImageProcessing.rgb_to_lab(img_batch)
        converted_batch = torch.clamp(converted_batch, 0.0, 1.0)
        return converted_batch
    
    @staticmethod
    def batch_L_ssim_convert(lab_img_batch):
        return lab_img_batch[:, 0, :, :].unsqueeze(1)
    
    @staticmethod
    def batch_hsv_convert(img_batch):
        # Batch calculate HSV values
        img_batch_hsv = ImageProcessing.rgb_to_hsv(img_batch)
        img_batch_hsv = torch.clamp(img_batch_hsv, 0.0, 1.0)
                                        
        img_batch_hue = 2*math.pi*img_batch_hsv[:, 0, :, :]
        img_batch_val = img_batch_hsv[:, 2, :, :]
        img_batch_sat = img_batch_hsv[:, 1, :, :]
                                              
        img_batch_1 = img_batch_val * img_batch_sat*torch.cos(img_batch_hue)
        img_batch_2 = img_batch_val * img_batch_sat*torch.sin(img_batch_hue)
                                              
        img_batch_hsv = torch.stack((img_batch_1, img_batch_2, img_batch_val), 1)
        return img_batch_hsv
        

    def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser):
        """Forward function for the CURL loss

        :param predicted_img_batch_high_res: 
        :param predicted_img_batch_high_res_rgb: 
        :param target_img_batch: Tensor of shape BxCxWxH
        :returns: value of loss function
        :rtype: float

        """
        target_img_batch_lab = self.batch_lab_convert(target_img_batch)
        predicted_img_batch_lab = self.batch_lab_convert(predicted_img_batch)
            
        target_img_batch_L_ssim = self.batch_L_ssim_convert(target_img_batch_lab)
        predicted_img_batch_L_ssim = self.batch_L_ssim_convert(predicted_img_batch_lab)
        
        target_img_batch_hsv = self.batch_hsv_convert(target_img_batch)
        predicted_img_batch_hsv = self.batch_hsv_convert(predicted_img_batch)
                                              
        # Calculate losses: batch
        l1_loss_value = F.l1_loss(predicted_img_batch_lab, target_img_batch_lab)
        rgb_loss_value = F.l1_loss(predicted_img_batch, target_img_batch)
        hsv_loss_value = F.l1_loss(predicted_img_batch_hsv, target_img_batch_hsv)                                              
        cosine_rgb_loss_value = (1.0 - torch.nn.functional.cosine_similarity(predicted_img_batch, 
                                                                            target_img_batch, dim=1).mean(dim=(1, 2))).mean()
        ssim_loss_value = (1.0 - self.compute_msssim(predicted_img_batch_L_ssim , target_img_batch_L_ssim)).mean()
        grad_reg = gradient_regulariser.mean()                

        curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
                     hsv_loss_value + 10*ssim_loss_value + 1e-6*grad_reg)/6

        return curl_loss


class CURLLayer(nn.Module):

    import torch.nn.functional as F
    
    converter_map = {
        ('rgb', 'lab'): ImageProcessing.rgb_to_lab,
        ('lab', 'rgb'): ImageProcessing.lab_to_rgb,
        ('rgb', 'hsv'): ImageProcessing.rgb_to_hsv,
        ('hsv', 'rgb'): ImageProcessing.hsv_to_rgb
        
    }

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation of class

        :param num_in_channels: number of input channels
        :param num_out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.make_init_network()

    def make_init_network(self):
        """ Initialise the CURL block layers

        :returns: N/A
        :rtype: N/A

        """
        self.lab_stack = ConvStack(conv_in=64, conv_out=64, curve_out=48, dropout=0.5)
        self.rgb_stack = ConvStack(conv_in=64, conv_out=64, curve_out=48, dropout=0.5)
        self.hsv_stack = ConvStack(conv_in=64, conv_out=64, curve_out=64, dropout=0.5)
        
    def convert(self, img, source, target):            
        img = torch.clamp(img, 0.0, 1.0)    
        converter = self.converter_map.get((source, target))
        
        if converter is None:
            ValueError("`source` and `target` must be one of: {}".format(self.converter_map.keys()))
        
        new_img = converter(img)
        new_img = torch.clamp(new_img, 0.0, 1.0)
        
        return new_img

    def forward(self, x):
        """Forward function for the CURL layer

        :param x: forward the data x through the network 
        :returns: Tensor representing the predicted image
        :rtype: Tensor

        """

        '''
        This function is where the magic happens :)
        '''
        x.contiguous()  # remove memory holes

        feat = x[:, 3:64, :, :]
        img = x[:, 0:3, :, :]

        torch.cuda.empty_cache()
        shape = x.shape

        # RGB -> LAB, modify LAB
        img_lab = self.convert(img, 'rgb', 'lab')
        feat_lab = torch.cat((feat, img_lab), 1)
        L = self.lab_stack(feat_lab)
        img_lab, gradient_regulariser_lab = ImageProcessing.adjust_lab(img_lab, L[:, 0:48])
        
        # LAB -> RGB, modify RGB
        img_rgb = self.convert(img_lab, 'lab', 'rgb')
        feat_rgb = torch.cat((feat, img_rgb), 1)
        R = self.rgb_stack(feat_rgb)
        img_rgb, gradient_regulariser_rgb = ImageProcessing.adjust_rgb(img_rgb, R[:, 0:48])
        
        # RGB -> HSV, modify HSV
        img_hsv = self.convert(img_rgb, 'rgb', 'hsv')
        feat_hsv = torch.cat((feat, img_hsv), 1)
        H = self.hsv_stack(feat_hsv)
        img_hsv, gradient_regulariser_hsv = ImageProcessing.adjust_hsv(img_hsv, H[:, 0:64])
        
        # HSV -> RGB, calculate residual and final image
        img_residual = self.convert(img_hsv, 'hsv', 'rgb')
        img = torch.clamp(img + img_residual, 0.0, 1.0)
        
        gradient_regulariser = gradient_regulariser_rgb + \
            gradient_regulariser_lab+gradient_regulariser_hsv

        return img, gradient_regulariser


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        return self.lrelu(self.conv(x))
    
class ConvStack(nn.Module):
    
    def __init__(self, conv_in=64, conv_out=64, curve_out=48, dropout=0.5):
        super(ConvStack, self).__init__()
        self.layer1 = ConvBlock(conv_in, conv_out)
        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = ConvBlock(conv_in, conv_out)
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = ConvBlock(conv_in, conv_out)
        self.layer6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer7 = ConvBlock(conv_in, conv_out)
        self.layer8 = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = torch.nn.Linear(conv_in, curve_out)
        
    def forward(self, feat_maps):
        x = self.layer1(feat_maps)
        del feat_maps
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        C = self.fc(x)
        
        return C


class CURLNet(nn.Module):

    def __init__(self):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CURLNet, self).__init__()
        self.tednet = rgb_ted.TEDModel()
        self.curllayer = CURLLayer()

    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        feat = self.tednet(img)
        img, gradient_regulariser = self.curllayer(feat)
        return img, gradient_regulariser

class CURLGlobalNet(nn.Module):

    def __init__(self):
        super(CURLGlobalNet, self).__init__()
        self.final_conv = nn.Conv2d(3, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)
        self.curllayer = CURLLayer()

    def forward(self, img):
        return self.curllayer(self.final_conv(self.refpad(img)))
