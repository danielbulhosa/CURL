# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 

'''
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math
import numpy as np
from torch.autograd import Variable
import torch

import matplotlib
import sys
matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)


class ImageProcessing(object):

    @staticmethod
    def rgb_to_lab(img, is_training=True):
        """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor

        """
        img = img.contiguous()

        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169,
             0.950227],  # B
        ]), requires_grad=False).cuda()

        img = torch.einsum('bcxy,ck->bkxy', img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1/0.950456, 1.0, 1/1.088754]), requires_grad=False).cuda().reshape(1, 3, 1, 1))

        epsilon = 6/29

        img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
            (torch.clamp(img, min=0.0001) **
             (1.0/3.0) * img.gt(epsilon**3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0,  500.0,    0.0],  # fx
                                                    # fy
                                                    [116.0, -500.0,  200.0],
                                                    # fz
                                                    [0.0,    0.0, -200.0],
                                                    ]), requires_grad=False).cuda()

        img = torch.einsum('bcxy,ck->bkxy', img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]).reshape(1, 3, 1, 1), requires_grad=False).cuda()

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img[:, 0, :, :] = img[:, 0, :, :]/100
        img[:, 1, :, :] = (img[:, 1, :, :]/110 + 1)/2
        img[:, 2, :, :] = (img[:, 2, :, :]/110 + 1)/2
        
        #img[(img != img).detach()] = 0  # This line causes memory error

        img = img.contiguous()
        return img.cuda()

    @staticmethod
    def lab_to_rgb(img, is_training=True):
        """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor
        """                
        img = img.contiguous()
        img_copy = img.clone()

        img_copy[:, 0, :, :] = img[:, 0, :, :] * 100
        img_copy[:, 1, :, :] = ((img[:, 1, :, :] * 2)-1)*110
        img_copy[:, 2, :, :] = ((img[:, 2, :, :] * 2)-1)*110

        img = img_copy.clone().cuda()
        del img_copy

        lab_to_fxfyfz = Variable(torch.FloatTensor([  # X Y Z
            [1/116.0, 1/116.0, 1/116.0],  # R
            [1/500.0, 0, 0],  # G
            [0, 0, -1/200.0],  # B
        ]), requires_grad=False).cuda()

        img = torch.einsum('bcxy,ck->bkxy', 
                           img + Variable(torch.FloatTensor([16.0, 0.0, 0.0]).cuda()).reshape(1, 3, 1, 1), 
                           lab_to_fxfyfz)

        epsilon = 6.0/29.0

        img = (((3.0 * epsilon**2 * (img-4.0/29.0)) * img.le(epsilon).float()) +
               ((torch.clamp(img, min=0.0001)**3.0) * img.gt(epsilon).float()))

        # denormalize for D65 white point
        img = torch.mul(img, Variable(
            torch.FloatTensor([0.950456, 1.0, 1.088754]).cuda().reshape(1, 3, 1, 1)))

        xyz_to_rgb = Variable(torch.FloatTensor([  # X Y Z
            [3.2404542, -0.9692660,  0.0556434],  # R
            [-1.5371385,  1.8760108, -0.2040259],  # G
            [-0.4985314,  0.0415560,  1.0572252],  # B
        ]), requires_grad=False).cuda()

        img = torch.einsum('bcxy,ck->bkxy', img, xyz_to_rgb)
        img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                        min=0.0001) ** (1/2.4) * 1.055) - 0.055) * img.gt(0.0031308).float()

        img = img.contiguous()
        #img[(img != img).detach()] = 0 # This line causes memory error
        
        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        """Move the image channels to the last dimensiion of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        img = ImageProcessing.normalise_image(np.array(Image.open(img_filepath)), normaliser)  # NB: imread normalises to 0-1
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                np.log10(max_intensity ** 2 /
                         ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        """Computes the SSIM for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        ssim_val = 0.0

        for i in range(0, num_images):
            imageA = ImageProcessing.swapimdims_3HW_HW3(
                image_batchA[i, 0:3, :, :])
            imageB = ImageProcessing.swapimdims_3HW_HW3(
                image_batchB[i, 0:3, :, :])
            ssim_val += ssim(imageA, imageB, data_range=imageA.max() - imageA.min(), multichannel=True,
                             gaussian_weights=True, win_size=11)

        return ssim_val / num_images

    @staticmethod
    def hsv_to_rgb(img):
        """Converts a HSV image to RGB
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: HSV image
        :returns: RGB image
        :rtype: Tensor

        """
        img=torch.clamp(img, 0.0, 1.0)
        
        m1 = 0
        m2 = (img[:, 2, :, :]*(1-img[:, 1, :, :])-img[:, 2, :, :])/60
        m3 = 0
        m4 = -1*m2
        m5 = 0

        r = img[:, 2, :, :]+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 60.0)*m1+torch.clamp(img[:, 0, :, :]*360-60, 0.0, 60.0)*m2+torch.clamp(
            img[:, 0, :, :]*360-120, 0.0, 120.0)*m3+torch.clamp(img[:, 0, :, :]*360-240, 0.0, 60.0)*m4+torch.clamp(img[:, 0, :, :]*360-300, 0.0, 60.0)*m5

        m1 = (img[:, 2, :, :]-img[:, 2, :, :]*(1-img[:, 1, :, :]))/60
        m2 = 0
        m3 = -1*m1
        m4 = 0

        g = img[:, 2, :, :]*(1-img[:, 1, :, :])+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 60.0)*m1+torch.clamp(img[:, 0, :, :]*360-60,
            0.0, 120.0)*m2+torch.clamp(img[:, 0, :, :]*360-180, 0.0, 60.0)*m3+torch.clamp(img[:, 0, :, :]*360-240, 0.0, 120.0)*m4

        m1 = 0
        m2 = (img[:, 2, :, :]-img[:, 2, :, :]*(1-img[:, 1, :, :]))/60
        m3 = 0
        m4 = -1*m2

        b = img[:, 2, :, :]*(1-img[:, 1, :, :])+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 120.0)*m1+torch.clamp(img[:, 0, :, :]*360 -
            120, 0.0, 60.0)*m2+torch.clamp(img[:, 0, :, :]*360-180, 0.0, 120.0)*m3+torch.clamp(img[:, 0, :, :]*360-300, 0.0, 60.0)*m4

        img = torch.stack((r, g, b), 1)
        #img[(img != img).detach()] = 0 # This causes memory error

        img = img.contiguous()
        img = torch.clamp(img, 0.0, 1.0)

        return img



    @staticmethod
    def rgb_to_hsv(img):
        """Converts an RGB image to HSV
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: RGB image
        :returns: HSV image
        :rtype: Tensor

        """
        # FIXME - complete refactor to taking batches
        img=torch.clamp(img, 10**(-9), 1.0)       

        # Permute channels c, x, y -> y, x, c
        img = img.permute(2, 1, 0)
        shape = img.shape

        img = img.contiguous()
        # Flatten along y, x dims, keep c
        img = img.view(-1, 3)

        # Take max and min along channel dim, take max values (0 index), not max idx (1 index)
        mx = torch.max(img, 1)[0]
        mn = torch.min(img, 1)[0]

        # Create ones 1D tensor with size y * x
        ones = Variable(torch.FloatTensor(
            torch.ones((img.shape[0])))).cuda()
        # Create zeros tensor with shape (x, y)
        zero = Variable(torch.FloatTensor(torch.zeros(shape[0:2]))).cuda()

        # Image back to shape y, x, c
        img = img.view(shape)
        
        # Chunk ones, max, and min tensors
        ones1, ones2 = torch.chunk(ones, 2, dim=0)
        mx1, mx2 = torch.chunk(mx, 2, dim=0)
        mn1, mn2 = torch.chunk(mn, 2, dim=0)

        # Add mx1, ones1, and mn1, same for 2. 
        df1 = torch.add(mx1, torch.mul(ones1*-1, mn1))
        df2 = torch.add(mx2, torch.mul(ones2*-1, mn2))

        # Cat results along 0 (chunk dim). Why did we chunk in the first place???
        df = torch.cat((df1, df2), 0)
        del df1, df2
        # Reshapes y * x size 1D tensor to (y, x) 2D shape tensor
        df = df.view(shape[0:2])+1e-10
        mx = mx.view(shape[0:2])

        img = img.cuda()
        df = df.cuda()
        mx = mx.cuda()

        # Extract each channel into its own (y, x) shape tensor
        g = img[:, :, 1].clone().cuda()
        b = img[:, :, 2].clone().cuda()
        r = img[:, :, 0].clone().cuda()

        # Shape: y, x, c
        img_copy = img.clone()
        
        # Transform color channel 0, (Hue?)
        img_copy[:, :, 0] = (((g-b)/df)*r.eq(mx).float() + (2.0+(b-r)/df)
                         * g.eq(mx).float() + (4.0+(r-g)/df)*b.eq(mx).float())
        img_copy[:, :, 0] = img_copy[:, :, 0]*60.0

        # New copy of image, move zero tensor to GPU
        zero = zero.cuda()
        img_copy2 = img_copy.clone()

        # More transformation of color channel 0, (Hue?)
        img_copy2[:, :, 0] = img_copy[:, :, 0].lt(zero).float(
        )*(img_copy[:, :, 0]+360) + img_copy[:, :, 0].ge(zero).float()*(img_copy[:, :, 0])

        img_copy2[:, :, 0] = img_copy2[:, :, 0]/360

        del img, r, g, b

        # Set saturation and value (S and V channels) with existing variables
        img_copy2[:, :, 1] = mx.ne(zero).float()*(df/mx) + \
            mx.eq(zero).float()*(zero)
        img_copy2[:, :, 2] = mx
        
        #img_copy2[(img_copy2 != img_copy2).detach()] = 0 # This line causes memory error

        img = img_copy2.clone()

        # Reshape y, x, c -> c, x, y
        img = img.permute(2, 1, 0)
        img = torch.clamp(img, 10**(-9), 1.0)

        return img

    
    @staticmethod
    def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out):
        """Applies a peicewise linear curve defined by a set of knot points to
        an image channel

        :param img: image to be adjusted
        :param C: predicted knot points of curve
        :returns: adjusted image
        :rtype: Tensor

        """
        slope = Variable(torch.zeros((C.shape[0]-1))).cuda()
        curve_steps = C.shape[0]-1

        '''
        Compute the slope of the line segments
        '''
        slope = C[1:]-C[0:-1]

        '''
        Compute the squared difference between slopes
        '''
        slope_sqr_diff += ((slope[1:]-slope[0:-1])**2).sum(0)

        '''
        Use predicted line segments to compute scaling factors for the channel
        '''
        steps = torch.arange(0, slope.shape[0]-1).cuda()
        image_channel = torch.unsqueeze(img[:, :,channel_in], -1) # expand dims to broadcast
        scale = C[0] + (slope[:-1] * (curve_steps * image_channel - steps)).sum(-1) # eq. 1
                
        img_copy = img.clone()
        img_copy[:, :, channel_out] = img[:, :, channel_out]*scale
        img_copy = torch.clamp(img_copy, 0.0, 1.0)
        
        return img_copy, slope_sqr_diff

    @staticmethod
    def adjust_hsv(img, S):
        """Adjust the HSV channels of a HSV image using learnt curves

        :param img: image to be adjusted 
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        S1, S2, S3, S4 = torch.chunk(S, 4, dim=0)
        S1, S2, S3, S4 = torch.exp(S1), torch.exp(S2), torch.exp(S3), torch.exp(S4)

        slope_sqr_diff = Variable(torch.zeros(1)*0.0).cuda()

        '''
        Adjust Hue channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, S1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Adjust Saturation channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1)
        
        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        #img[(img != img).detach()] = 0 # This line causes memory error

        img = img.permute(2, 1, 0)
        img = img.contiguous()
        
        return img, slope_sqr_diff

    @staticmethod
    def adjust_rgb(img, R):
        """Adjust the RGB channels of a RGB image using learnt curves

        :param img: image to be adjusted 
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        '''
        Extract the parameters of the three curves
        '''
        R1, R2, R3 = torch.chunk(R, 3, dim=0)
        R1, R2, R3 = torch.exp(R1), torch.exp(R2), torch.exp(R3)

        '''
        Apply the curve to the R channel 
        '''
        slope_sqr_diff = Variable(torch.zeros(1)*0.0).cuda()

        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, R1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Apply the curve to the G channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Apply the curve to the B channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        #img[(img != img).detach()] = 0 # This line causes memory error

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_lab(img, L):
        """Adjusts the image in LAB space using the predicted curves

        :param img: Image tensor
        :param L: Predicited curve parameters for LAB channels
        :returns: adjust image, and regularisation parameter
        :rtype: Tensor, float

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        '''
        Extract predicted parameters for each L,a,b curve
        '''
        L1, L2, L3 = torch.chunk(L, 3, dim=0)
        L1, L2, L3 = torch.exp(L1), torch.exp(L2), torch.exp(L3)

        slope_sqr_diff = Variable(torch.zeros(1)*0.0).cuda()

        '''
        Apply the curve to the L channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, L1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Now do the same for the a channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Now do the same for the b channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        #img[(img != img).detach()] = 0 # This line causes memory error

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff
