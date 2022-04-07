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
import metric
import colors
import curves
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as trans
import operator as op
import timm
from functools import reduce

np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CURLLoss(nn.Module):
    
    def __init__(self, ssim_window_size=5, num_channel=1):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLoss, self).__init__()
        self.ssim_window_size = ssim_window_size
        self.num_channel = num_channel
        self.msssim_layer = metric.MSSSIMMetric(num_channel=num_channel)
        self.rgb2lab = colors.RGB2LAB()
        self.rgb2hsv = colors.RGB2HSV()
    
    def batch_lab_convert(self, img_batch):
        converted_batch = self.rgb2lab(img_batch)
        converted_batch = torch.clamp(converted_batch, 0.0, 1.0)
        return converted_batch
    
    @staticmethod
    def batch_L_ssim_convert(lab_img_batch):
        return lab_img_batch[:, 0, :, :].unsqueeze(1)
    
    def batch_hsv_convert(self, img_batch):
        # Batch calculate HSV values
        img_batch_hsv = self.rgb2hsv(img_batch)
        img_batch_hsv = torch.clamp(img_batch_hsv, 0.0, 1.0)
                                        
        img_batch_hue = 2*math.pi*img_batch_hsv[:, 0, :, :]
        img_batch_val = img_batch_hsv[:, 2, :, :]
        img_batch_sat = img_batch_hsv[:, 1, :, :]
                                              
        img_batch_1 = img_batch_val * img_batch_sat*torch.cos(img_batch_hue)
        img_batch_2 = img_batch_val * img_batch_sat*torch.sin(img_batch_hue)
                                              
        img_batch_hsv = torch.stack((img_batch_1, img_batch_2, img_batch_val), 1)
        return img_batch_hsv
        

    def forward(self, predicted_img_batch, target_img_batch, mask):
        """Forward function for the CURL loss

        :param predicted_img_batch_high_res: 
        :param predicted_img_batch_high_res_rgb: 
        :param target_img_batch: Tensor of shape BxCxWxH
        :returns: value of loss function
        :rtype: float

        """
        # Apply mask for purposes of loss function
        unmasked_pixels = predicted_img_batch.shape[1] * mask.sum()  # Multiply by number of channels
        predicted_img_batch, target_img_batch = predicted_img_batch * mask, target_img_batch * mask
        
        rgb_loss_value = F.l1_loss(predicted_img_batch, target_img_batch, reduction='sum')/unmasked_pixels
        
        # Cosine similarity for #000000 (black) should be 1. Accomplishing this by
        # finding mask values and manually setting similarity to 1. 
        base_cos_sim = torch.nn.functional.cosine_similarity(predicted_img_batch, target_img_batch, dim=1)
        cosine_rgb_loss_value = (1.0 - (base_cos_sim + torch.logical_not(mask)).mean(dim=(1, 2))).mean()
        
        target_img_batch_lab = self.batch_lab_convert(target_img_batch)
        predicted_img_batch_lab = self.batch_lab_convert(predicted_img_batch)
        l1_loss_value = F.l1_loss(predicted_img_batch_lab, target_img_batch_lab, reduction='sum')/unmasked_pixels
            
        target_img_batch_L_ssim = self.batch_L_ssim_convert(target_img_batch_lab)
        predicted_img_batch_L_ssim = self.batch_L_ssim_convert(predicted_img_batch_lab)
        ssim_loss_value = (1.0 - self.msssim_layer(predicted_img_batch_L_ssim , target_img_batch_L_ssim)).mean()
        
        target_img_batch_hsv = self.batch_hsv_convert(target_img_batch)
        predicted_img_batch_hsv = self.batch_hsv_convert(predicted_img_batch)
        hsv_loss_value = F.l1_loss(predicted_img_batch_hsv, target_img_batch_hsv, reduction='sum')/unmasked_pixels

        curl_loss = (rgb_loss_value +
                     cosine_rgb_loss_value +
                     l1_loss_value +
                     hsv_loss_value +
                     10*ssim_loss_value
                    )/5

        return curl_loss


class CURLLayer(nn.Module):

    def __init__(self, num_lab_points=48, num_rgb_points=48, 
                 num_hsv_points=64):
        
        super(CURLLayer, self).__init__()

        self.num_lab_points = num_lab_points
        self.num_rgb_points = num_rgb_points
        self.num_hsv_points = num_hsv_points
        self.rgb2lab = colors.RGB2LAB()
        self.lab2rgb = colors.LAB2RGB()
        self.rgb2hsv = colors.RGB2HSV()
        self.hsv2rgb = colors.HSV2RGB()


    def forward(self, img, mask, L, R, H):
        """Forward function for the CURL layer

        :param x: forward the data x through the network 
        :returns: Tensor representing the predicted image
        :rtype: Tensor

        """

        '''
        This function is where the magic happens :)
        '''        

        # RGB -> LAB, modify LAB
        img_lab = self.rgb2lab(img)
        feat_lab = torch.cat((feat, img_lab), 1)
        img_lab, gradient_regulariser_lab = curves.adjust_lab(img_lab, L[:, :self.num_lab_points])
        img_lab = img_lab * mask
        
        # LAB -> RGB, modify RGB
        img_rgb = self.lab2rgb(img_lab)
        feat_rgb = torch.cat((feat, img_rgb), 1)
        img_rgb, gradient_regulariser_rgb = curves.adjust_rgb(img_rgb, R[:, :self.num_rgb_points])
        img_rgb = img_rgb * mask
        
        # RGB -> HSV, modify HSV
        img_hsv = self.rgb2hsv(img_rgb)
        feat_hsv = torch.cat((feat, img_hsv), 1)
        img_hsv, gradient_regulariser_hsv = curves.adjust_hsv(img_hsv, H[:, :self.num_hsv_points])
        img_hsv = img_hsv * mask
        
        # HSV -> RGB, calculate residual and final image
        img_residual = self.hsv2rgb(img_hsv)
        img = torch.clamp(img + img_residual, 0.0, 1.0) * mask
        
        gradient_regulariser = (gradient_regulariser_rgb +
                                gradient_regulariser_lab +
                                gradient_regulariser_hsv)

        return img, gradient_regulariser
    

class GCURLNet(nn.Module):
        
    def __init__(self, num_lab_points=48, num_rgb_points=48, num_hsv_points=64):
        super(GCURLNet, self).__init__()
        self.num_lab_points = num_lab_points
        self.num_rgb_points = num_rgb_points
        self.num_hsv_points = num_hsv_points
        self.curve_break_1 = num_lab_points
        self.curve_break_2 = num_lab_points + num_rgb_points
        
        self.backbone = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.backbone.classifier = nn.Sequential(nn.Linear(in_features=1792, 
                                                           out_features=self.num_spaces*self.num_channels*self.num_coeffs)
                                               )
        self.curllayer = CURLLayer(self.num_lab_points, self.num_rgb_points, self.num_hsv_points)
    
    def forward(self, img, mask, L, R, H):
        curves = self.backbone(img)
        L, R, H = curves[:, :self.curve_break_1], \
                  curves[:, self.curve_break_1:self.curve_break_2], \
                  curves[:, self.curve_break_2:]
        
        img, gradient_regulariser = self.curllayer(img, mask, L, R, H)
    
        return img, gradient_regulariser

        
class ChannelPolyLayer(nn.Module):
    
    def __init__(self, degree=3, num_variables=3, num_out=None):
        assert degree >= 0 and type(degree) == int, "`degree` must be non-negative integer"
        assert num_variables >= 0 and type(num_variables) == int, "`num_variables` must be non-negative integer"
        
        super(ChannelPolyLayer, self).__init__()
        self.degree = degree
        self.num_variables = num_variables
        self.num_out = self.num_variables if num_out is None else num_out
        self.num_coeffs = ChannelPolyLayer.ncr(num_variables + degree, degree)
        self.powers = nn.Parameter(torch.Tensor(list(ChannelPolyLayer.generate_powers(degree, num_variables))),
                                   requires_grad=False)
        
        assert self.powers.shape[0] == self.num_coeffs,\
            "Number of coefficients and powers should match"
     
    @staticmethod
    def generate_powers(order, n_variables):
        """
        Find the exponents of a multivariate polynomial expression of order
        `order` and `n_variable` number of variables.
        From: https://stackoverflow.com/questions/4913902/optimize-generator-for-multivariate-polynomial-exponents
        """
        pattern = [0] * n_variables
        yield tuple(pattern)
        for current_sum in range(1, order+1):
            pattern[0] = current_sum
            yield tuple(pattern)
            while pattern[-1] < current_sum:
                for i in range(2, n_variables + 1):
                    if 0 < pattern[n_variables - i]:
                        pattern[n_variables - i] -= 1
                        if 2 < i:
                            pattern[n_variables - i + 1] = 1 + pattern[-1]
                            pattern[-1] = 0
                        else:
                            pattern[-1] += 1
                        break
                yield tuple(pattern)
            pattern[-1] = 0

    @staticmethod
    def generate_poly_string(img_name, coeff_name, order, n_variables):
        poly_terms = []

        for term_num, powers in enumerate(ChannelPolyLayer.generate_powers(order, n_variables)):
            poly_terms.append('{}[:, {}]'.format(coeff_name, term_num))
            for idx, power in enumerate(powers):
                if power == 0:
                    continue
                elif power == 1:
                    poly_terms[term_num] += "*{}[:, {}]".format(img_name, idx)
                else:
                    poly_terms[term_num] += "*({}[:, {}]**{})".format(img_name, idx, power)

        return ' + '.join(poly_terms)

    @staticmethod
    def generate_poly_terms(img_us_name, order, n_variables):
        poly_terms = []

        for term_num, powers in enumerate(ChannelPolyLayer.generate_powers(order, n_variables)):
            monomials = []

            for idx, power in enumerate(powers):
                if power == 0:
                    continue
                elif power == 1:
                    monomials.append("{}[:, {}]".format(img_us_name, idx))
                else:
                    monomials.append("({}[:, {}]**{})".format(img_us_name, idx, power))

            if len(monomials) == 0:
                poly_term = '(1.0 + {}[:, 0] * 0.0)'.format(img_us_name)
            else:
                poly_term = '*'.join(monomials)

            poly_terms.append(poly_term)

        return 'torch.cat([' + ', '.join(poly_terms) + '], dim=-1)'

    @staticmethod
    def ncr(n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom  # or / in Python 2
        
    def forward(self, img, coeffs):
        assert img.shape[1] == self.num_variables, \
        "There should be a polynomial variable per channel"
        assert coeffs.shape[2] == self.num_coeffs,\
            "For degree {} and number of variables {} the number of required " \
            "coefficients (coeffs.shape[1]) is {}".format(degree, num_variables, self.num_coeffs)
        assert len(coeffs.shape) == 3,\
            "Coefficients be 2D tensor of shape (batch size, num coefficients)"
        
        """
        Dims (batch size, out channels, x, y, num coeffs)
        Creates polynomial terms, e.g. for degree 3 polynomial with
        2 variables polynomial_terms.shape[-1] equals 10 with the 
        the last dimension of this tensor equal to 
        (1, x, y, x^2, xy, y^2, x^3, x^2y^1, x^1y^2, y^3)
        Note that in our case our polynomial has variables equal to
        the number of channels/colors in the image, so x, y here should
        not be interpreted as spatial coordinates.
        
        Note we can have different input channels `self.num_variables`
        and output channels `self.num_out`. This allows us to for example
        output 3 color channels from a 5 dimensional image input where 3
        channels are colors and the other 2 are spatial coordinates.
        """
        img_us = torch.unsqueeze(img, dim=0)
        pwrs_rs = self.powers.reshape(self.num_coeffs, 1, self.num_variables, 1, 1)
        poly_terms = torch.permute(torch.pow(img_us, pwrs_rs), [1, 2, 3, 4, 0]).prod(dim=1)
        
        """
        Dims (batch size, out channels, x, y)
        Continuing the example above, we take the polynomial terms,
        multiply by their learned coefficients, and sum to get
        a + bx + cy dx^2 + exy + fy^2 + gx^3 + hx^2y^1 + ix^1y^2 + jy^3)
        where coeffs is the tensor containing the predicted 
        (a, b, c, d, e, f, g, h, i, j) for each image in the minibatch.
        """
        
        return (coeffs.reshape(img.shape[0], self.num_out, 1, 1, self.num_coeffs) * \
                torch.unsqueeze(poly_terms, dim=1)).sum(dim=-1)


class Deg4MobilePolyLayer(nn.Module):

    def __init__(self):
        super(Deg4MobilePolyLayer, self).__init__()
        self.num_coeffs = 126
        # Need to be able to load state from ChannelPolyLayer
        self.powers = nn.Parameter(torch.Tensor(list(ChannelPolyLayer.generate_powers(4, 5))),
                                   requires_grad=False)

    @staticmethod
    def poly_terms(img_us):
        # Generated with `ChannelPolyLayer.generate_poly_terms('img_us', 4, 5)`
        poly_terms = torch.cat([(1.0 + img_us[:, 0] * 0.0), img_us[:, 0], img_us[:, 1], img_us[:, 2], img_us[:, 3],
                                img_us[:, 4], (img_us[:, 0]**2), img_us[:, 0]*img_us[:, 1], img_us[:, 0]*img_us[:, 2],
                                img_us[:, 0]*img_us[:, 3], img_us[:, 0]*img_us[:, 4], (img_us[:, 1]**2), img_us[:, 1]*img_us[:, 2],
                                img_us[:, 1]*img_us[:, 3], img_us[:, 1]*img_us[:, 4], (img_us[:, 2]**2), img_us[:, 2]*img_us[:, 3],
                                img_us[:, 2]*img_us[:, 4], (img_us[:, 3]**2), img_us[:, 3]*img_us[:, 4], (img_us[:, 4]**2),
                                (img_us[:, 0]**3), (img_us[:, 0]**2)*img_us[:, 1], (img_us[:, 0]**2)*img_us[:, 2],
                                (img_us[:, 0]**2)*img_us[:, 3], (img_us[:, 0]**2)*img_us[:, 4], img_us[:, 0]*(img_us[:, 1]**2),
                                img_us[:, 0]*img_us[:, 1]*img_us[:, 2], img_us[:, 0]*img_us[:, 1]*img_us[:, 3],
                                img_us[:, 0]*img_us[:, 1]*img_us[:, 4], img_us[:, 0]*(img_us[:, 2]**2),
                                img_us[:, 0]*img_us[:, 2]*img_us[:, 3], img_us[:, 0]*img_us[:, 2]*img_us[:, 4],
                                img_us[:, 0]*(img_us[:, 3]**2), img_us[:, 0]*img_us[:, 3]*img_us[:, 4],
                                img_us[:, 0]*(img_us[:, 4]**2), (img_us[:, 1]**3), (img_us[:, 1]**2)*img_us[:, 2],
                                (img_us[:, 1]**2)*img_us[:, 3], (img_us[:, 1]**2)*img_us[:, 4], img_us[:, 1]*(img_us[:, 2]**2),
                                img_us[:, 1]*img_us[:, 2]*img_us[:, 3], img_us[:, 1]*img_us[:, 2]*img_us[:, 4],
                                img_us[:, 1]*(img_us[:, 3]**2), img_us[:, 1]*img_us[:, 3]*img_us[:, 4],
                                img_us[:, 1]*(img_us[:, 4]**2), (img_us[:, 2]**3), (img_us[:, 2]**2)*img_us[:, 3],
                                (img_us[:, 2]**2)*img_us[:, 4], img_us[:, 2]*(img_us[:, 3]**2), img_us[:, 2]*img_us[:, 3]*img_us[:, 4],
                                img_us[:, 2]*(img_us[:, 4]**2), (img_us[:, 3]**3), (img_us[:, 3]**2)*img_us[:, 4],
                                img_us[:, 3]*(img_us[:, 4]**2), (img_us[:, 4]**3), (img_us[:, 0]**4), (img_us[:, 0]**3)*img_us[:, 1],
                                (img_us[:, 0]**3)*img_us[:, 2], (img_us[:, 0]**3)*img_us[:, 3], (img_us[:, 0]**3)*img_us[:, 4],
                                (img_us[:, 0]**2)*(img_us[:, 1]**2), (img_us[:, 0]**2)*img_us[:, 1]*img_us[:, 2],
                                (img_us[:, 0]**2)*img_us[:, 1]*img_us[:, 3], (img_us[:, 0]**2)*img_us[:, 1]*img_us[:, 4],
                                (img_us[:, 0]**2)*(img_us[:, 2]**2), (img_us[:, 0]**2)*img_us[:, 2]*img_us[:, 3],
                                (img_us[:, 0]**2)*img_us[:, 2]*img_us[:, 4], (img_us[:, 0]**2)*(img_us[:, 3]**2),
                                (img_us[:, 0]**2)*img_us[:, 3]*img_us[:, 4], (img_us[:, 0]**2)*(img_us[:, 4]**2),
                                img_us[:, 0]*(img_us[:, 1]**3), img_us[:, 0]*(img_us[:, 1]**2)*img_us[:, 2],
                                img_us[:, 0]*(img_us[:, 1]**2)*img_us[:, 3], img_us[:, 0]*(img_us[:, 1]**2)*img_us[:, 4],
                                img_us[:, 0]*img_us[:, 1]*(img_us[:, 2]**2), img_us[:, 0]*img_us[:, 1]*img_us[:, 2]*img_us[:, 3],
                                img_us[:, 0]*img_us[:, 1]*img_us[:, 2]*img_us[:, 4], img_us[:, 0]*img_us[:, 1]*(img_us[:, 3]**2),
                                img_us[:, 0]*img_us[:, 1]*img_us[:, 3]*img_us[:, 4], img_us[:, 0]*img_us[:, 1]*(img_us[:, 4]**2),
                                img_us[:, 0]*(img_us[:, 2]**3), img_us[:, 0]*(img_us[:, 2]**2)*img_us[:, 3],
                                img_us[:, 0]*(img_us[:, 2]**2)*img_us[:, 4], img_us[:, 0]*img_us[:, 2]*(img_us[:, 3]**2),
                                img_us[:, 0]*img_us[:, 2]*img_us[:, 3]*img_us[:, 4], img_us[:, 0]*img_us[:, 2]*(img_us[:, 4]**2),
                                img_us[:, 0]*(img_us[:, 3]**3), img_us[:, 0]*(img_us[:, 3]**2)*img_us[:, 4], img_us[:, 0]*img_us[:, 3]*(img_us[:, 4]**2),
                                img_us[:, 0]*(img_us[:, 4]**3), (img_us[:, 1]**4), (img_us[:, 1]**3)*img_us[:, 2], (img_us[:, 1]**3)*img_us[:, 3],
                                (img_us[:, 1]**3)*img_us[:, 4], (img_us[:, 1]**2)*(img_us[:, 2]**2), (img_us[:, 1]**2)*img_us[:, 2]*img_us[:, 3],
                                (img_us[:, 1]**2)*img_us[:, 2]*img_us[:, 4], (img_us[:, 1]**2)*(img_us[:, 3]**2), (img_us[:, 1]**2)*img_us[:, 3]*img_us[:, 4],
                                (img_us[:, 1]**2)*(img_us[:, 4]**2), img_us[:, 1]*(img_us[:, 2]**3), img_us[:, 1]*(img_us[:, 2]**2)*img_us[:, 3],
                                img_us[:, 1]*(img_us[:, 2]**2)*img_us[:, 4], img_us[:, 1]*img_us[:, 2]*(img_us[:, 3]**2),
                                img_us[:, 1]*img_us[:, 2]*img_us[:, 3]*img_us[:, 4], img_us[:, 1]*img_us[:, 2]*(img_us[:, 4]**2),
                                img_us[:, 1]*(img_us[:, 3]**3), img_us[:, 1]*(img_us[:, 3]**2)*img_us[:, 4],
                                img_us[:, 1]*img_us[:, 3]*(img_us[:, 4]**2), img_us[:, 1]*(img_us[:, 4]**3), (img_us[:, 2]**4),
                                (img_us[:, 2]**3)*img_us[:, 3], (img_us[:, 2]**3)*img_us[:, 4], (img_us[:, 2]**2)*(img_us[:, 3]**2),
                                (img_us[:, 2]**2)*img_us[:, 3]*img_us[:, 4], (img_us[:, 2]**2)*(img_us[:, 4]**2),
                                img_us[:, 2]*(img_us[:, 3]**3), img_us[:, 2]*(img_us[:, 3]**2)*img_us[:, 4],
                                img_us[:, 2]*img_us[:, 3]*(img_us[:, 4]**2), img_us[:, 2]*(img_us[:, 4]**3),
                                (img_us[:, 3]**4), (img_us[:, 3]**3)*img_us[:, 4], (img_us[:, 3]**2)*(img_us[:, 4]**2),
                                img_us[:, 3]*(img_us[:, 4]**3), (img_us[:, 4]**4)], dim=-1)

        return poly_terms

    def forward(self, img, coeffs):
        """
        Equivalent to ChannelPolyLayer(degree=4, num_variables=5, num_out=3).forward
        We just explicitly write out poly_terms for Core ML compatibility

        Note that doing torch.cat + tensor.sum instead of using the formula generate_poly_string
        gives us an output that is exactly equal to that of ChannelPolyLayer.forward
        down to floating point precision. If we used the formula from generate_poly_string
        we'd get floating point discrepancies that impact the fidelity of the output. This is
        because tensor.sum and the + operator on tensors yield different results when summing
        many terms down to floating point precision.
        """
        img_us = torch.unsqueeze(img, dim=-1)
        poly_terms = Deg4MobilePolyLayer.poly_terms(img_us)

        return (coeffs.reshape(img.shape[0], 3, 1, 1, self.num_coeffs) * \
                torch.unsqueeze(poly_terms, dim=1)).sum(dim=-1)


class PolyRegNet(nn.Module):
    
    def __init__(self, num_channels=3, polynomial_order=4):
        super(PolyRegNet, self).__init__()
        self.num_channels = num_channels
        self.order = polynomial_order
        self.polylayer = ChannelPolyLayer(degree=self.order, num_variables=self.num_channels)
        self.num_coeffs = self.polylayer.num_coeffs
        
        self.backbone = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.backbone.classifier = nn.Linear(in_features=1792, 
                                             out_features=self.num_channels*self.num_coeffs)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, img, mask):
        coeffs = self.backbone(img).reshape(img.shape[0], self.num_channels, self.num_coeffs)
        final_img = self.sigmoid(self.polylayer(img, coeffs)) * mask
        
        return final_img
    

class TriSpaceRegNet(nn.Module):
    
    def __init__(self, polynomial_order=4, spatial=False, 
                 max_resolution=10000, is_train=True,
                 use_sync_bn=False, polylayer=None):
        super(TriSpaceRegNet, self).__init__()
        self.num_channels = 3
        self.num_spaces = 3
        self.num_in = self.num_channels + 2 * spatial
        self.order = polynomial_order
        self.is_train = is_train
        self.max_resolution = max_resolution
        self.polylayer = polylayer if polylayer is not None else ChannelPolyLayer(degree=self.order,
                                                                                  num_variables=self.num_in,
                                                                                  num_out=self.num_channels)
        self.num_coeffs = self.polylayer.num_coeffs

        self.backbone = timm.create_model('efficientnetv2_rw_t', pretrained=True)
        if use_sync_bn:
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.backbone.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=1024), # 1280 in for b0
                                                 nn.Linear(in_features=1024, out_features=512),
                                                 nn.Linear(in_features=512, out_features=512),
                                                 nn.Linear(in_features=512, out_features=self.num_spaces*self.num_channels*self.num_coeffs)
                                               )
        self.rgb2lab = colors.RGB2LAB()
        self.lab2rgb = colors.LAB2RGB()
        self.rgb2hsv = colors.RGB2HSV()
        self.hsv2rgb = colors.HSV2RGB()
        
        self.sigmoid = nn.Sigmoid()
        
        # We found that making the properties parameter values conditional agrees with
        # DataParallel, but if we rely on pointing methods instead of objects DataParallel
        # doesn't like that when the method uses parameters.
        if not spatial:
            self.x = torch.zeros(1, 0, 1, self.max_resolution)
            self.y = torch.zeros(1, 0, self.max_resolution, 1)
        else:
            self.x = torch.arange(0, self.max_resolution).reshape(1, 1, 1, self.max_resolution)
            self.y = torch.arange(0, self.max_resolution).reshape(1, 1, self.max_resolution, 1)
            
        self.x = torch.nn.Parameter(self.x, requires_grad=False)
        self.y = torch.nn.Parameter(self.y, requires_grad=False)

    def cat_coords(self, img):
        """
        Concatenates actual coordinate values to channel dimension
        """
        batch_size, channels, height, width = img.shape
        assert height <= self.max_resolution and width <= self.max_resolution, \
            "img width and height must be less than `max_resolution`, set for instance to: {}".format(self.max_resolution)
        
        zeros = img[:, 0:1] * 0.0
        x = zeros + self.x[:, :, :, :width]/width
        y = zeros + self.y[:, :, :height, :]/height
        return torch.cat([img, x, y], dim=1)
    
    def generate_image(self, img, R, L, H):
        img_rgb = self.cat_coords(img)
        img_lab = self.cat_coords(self.rgb2lab(img))
        img_hsv = self.cat_coords(self.rgb2hsv(img))
        
        rgb_res = self.sigmoid(self.polylayer(img_rgb, R))
        lab_res = self.lab2rgb(self.sigmoid(self.polylayer(img_lab, L)))
        hsv_res = self.hsv2rgb(self.sigmoid(self.polylayer(img_hsv, H)))
        
        # Modify all pixels
        rgb_res = 2 * (rgb_res - 0.5)
        lab_res = 2 * (lab_res - 0.5)
        hsv_res = 2 * (hsv_res - 0.5)
        
        residual = rgb_res + lab_res + hsv_res

        return torch.clamp(img + residual, 0.0, 1.0)
    
    def generate_coefficients(self, img, mask):
        coeffs = self.backbone(img * mask).reshape(img.shape[0], self.num_spaces, 
                                                   self.num_channels, self.num_coeffs)
        
        R, L, H = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
        return R, L, H
    
    def forward(self, img, mask):
        return self.generate_coefficients(img, mask)
