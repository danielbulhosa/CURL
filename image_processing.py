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
import numpy as np
from torch.autograd import Variable
import torch

import matplotlib
import sys
matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)

    
rgb_to_xyz = Variable(torch.FloatTensor(
    [  # X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ]), requires_grad=False).cuda()

fxfyfz_to_lab = Variable(torch.FloatTensor(
    [
        [0.0,  500.0,    0.0],  # fx
        [116.0, -500.0,  200.0],  # fy
        [0.0,    0.0, -200.0],  # fz
    ]), requires_grad=False).cuda()

lab_to_fxfyfz = Variable(torch.FloatTensor(
    [   # X       Y         Z
        [1/116.0, 1/116.0, 1/116.0],  # R
        [1/500.0, 0, 0],  # G
        [0, 0, -1/200.0],  # B
    ]), requires_grad=False).cuda()

xyz_to_rgb = Variable(torch.FloatTensor(
    [   # X          Y           Z
        [3.2404542, -0.9692660,  0.0556434],  # R
        [-1.5371385,  1.8760108, -0.2040259],  # G
        [-0.4985314,  0.0415560,  1.0572252],  # B
    ]), requires_grad=False).cuda()

lab_to_fxfyfz_offset = Variable(torch.FloatTensor([16.0, 0.0, 0.0]), requires_grad=False).reshape(1, 3, 1, 1).cuda()
xyz_to_rgb_mult = Variable(torch.FloatTensor([0.950456, 1.0, 1.088754]), requires_grad=False).reshape(1, 3, 1, 1).cuda()


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

    img = torch.einsum('bcxy,ck->bkxy', img, ImageProcessing.rgb_to_xyz)
    img = torch.mul(img, 1/ImageProcessing.xyz_to_rgb_mult)

    epsilon = 6/29

    img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
        (torch.clamp(img, min=0.0001) **
         (1.0/3.0) * img.gt(epsilon**3).float())

    img = torch.einsum('bcxy,ck->bkxy', img, ImageProcessing.fxfyfz_to_lab) - ImageProcessing.lab_to_fxfyfz_offset

    '''
    L_chan: black and white with input range [0, 100]
    a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
    [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
    '''
    img[:, 0, :, :] = img[:, 0, :, :]/100
    img[:, 1, :, :] = (img[:, 1, :, :]/110 + 1)/2
    img[:, 2, :, :] = (img[:, 2, :, :]/110 + 1)/2

    img = img.contiguous()
    return img


def lab_to_rgb(img, is_training=True):
    """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    :param img: image to be adjusted
    :returns: adjusted image
    :rtype: Tensor
    """                
    img = img.contiguous()

    img[:, 0, :, :] = img[:, 0, :, :] * 100
    img[:, 1, :, :] = ((img[:, 1, :, :] * 2)-1)*110
    img[:, 2, :, :] = ((img[:, 2, :, :] * 2)-1)*110

    img = torch.einsum('bcxy,ck->bkxy', 
                       img + ImageProcessing.lab_to_fxfyfz_offset, 
                       ImageProcessing.lab_to_fxfyfz)

    epsilon = 6.0/29.0

    img = (((3.0 * epsilon**2 * (img-4.0/29.0)) * img.le(epsilon).float()) +
           ((torch.clamp(img, min=0.0001)**3.0) * img.gt(epsilon).float()))

    # denormalize for D65 white point
    img = torch.mul(img, ImageProcessing.xyz_to_rgb_mult)

    img = torch.einsum('bcxy,ck->bkxy', img, ImageProcessing.xyz_to_rgb)
    img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                    min=0.0001) ** (1/2.4) * 1.055) - 0.055) * img.gt(0.0031308).float()

    img = img.contiguous()

    return img


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


def load_image(img_filepath, normaliser):
    """Loads an image from file as a numpy multi-dimensional array

    :param img_filepath: filepath to the image
    :returns: image as a multi-dimensional numpy array
    :rtype: multi-dimensional numpy array

    """
    img = ImageProcessing.normalise_image(np.array(Image.open(img_filepath)), normaliser)  # NB: imread normalises to 0-1
    return img


def normalise_image(img, normaliser):
    """Normalises image data to be a float between 0 and 1

    :param img: Image as a numpy multi-dimensional image array
    :returns: Normalised image as a numpy multi-dimensional image array
    :rtype: Numpy array

    """
    img = img.astype('float32') / normaliser
    return img


def compute_mse(original_batch, result_batch):
    """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

    :param original: input RGB image as a numpy array
    :param result: target RGB image as a numpy array
    :returns: the mean squared error between the input and target images
    :rtype: float

    """
    mask = torch.logical_not(torch.logical_and((0 == original).all(dim=1),
                                       (0 == result).all(dim=1)))
    return ((original - result) ** 2).sum(dims=(1,2,3))/mask.sum(dims=(1,2))


def compute_psnr(image_batchA, image_batchB, max_intensity):
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
                                ImageProcessing.compute_mse(image_batchA, image_batchB))

    # Take average over batch dimension
    return psnr_val.mean()


def create_window(window_size, num_channel):
    """Window creation function for SSIM metric. Gaussian weights are applied to the window.
    Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

    :param window_size: size of the window to compute statistics
    :param num_channel: number of channels
    :returns: Tensor of shape Cx1xWindow_sizexWindow_size
    :rtype: Tensor

    """
    _1D_window = ImageProcessing.gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        num_channel, 1, window_size, window_size).contiguous())
    return window


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


# Default values for ssim methods
window_size = 11
channels = 3
gaussian_window = ImageProcessing.create_window(window_size, channels)


def compute_ssim(img1, img2, window=ImageProcessing.gaussian_window, 
                 ssim_window_size=ImageProcessing.window_size, 
                 num_channel=ImageProcessing.channels):
    """Computes the structural similarity index between two images. This function is differentiable.
    Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    Note ssim does not assume a color space (https://arxiv.org/pdf/2006.13846.pdf)

    :param img1: image Tensor BxCxHxW
    :param img2: image Tensor BxCxHxW
    :returns: mean SSIM
    :rtype: float

    """
    window = window.type_as(img1)

    mu1 = F.conv2d(
        img1, window, padding=ssim_window_size // 2, groups=num_channel)
    mu2 = F.conv2d(
        img2, window, padding=ssim_window_size // 2, groups=num_channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=ssim_window_size // 2, groups=num_channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=ssim_window_size // 2, groups=num_channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=ssim_window_size // 2, groups=num_channel) - mu1_mu2

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


def compute_msssim(img1, img2, window=ImageProcessing.gaussian_window, 
                   ssim_window_size=ImageProcessing.window_size, 
                   num_channel=ImageProcessing.channels):
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
        ssim, cs = ImageProcessing.compute_ssim(img1, img2, window, ssim_window_size, num_channel)

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
    del m1, m2, m3, m4, m5

    m1 = (img[:, 2, :, :]-img[:, 2, :, :]*(1-img[:, 1, :, :]))/60
    m2 = 0
    m3 = -1*m1
    m4 = 0

    g = img[:, 2, :, :]*(1-img[:, 1, :, :])+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 60.0)*m1+torch.clamp(img[:, 0, :, :]*360-60,
        0.0, 120.0)*m2+torch.clamp(img[:, 0, :, :]*360-180, 0.0, 60.0)*m3+torch.clamp(img[:, 0, :, :]*360-240, 0.0, 120.0)*m4
    del m1, m2, m3, m4

    m1 = 0
    m2 = (img[:, 2, :, :]-img[:, 2, :, :]*(1-img[:, 1, :, :]))/60
    m3 = 0
    m4 = -1*m2

    b = img[:, 2, :, :]*(1-img[:, 1, :, :])+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 120.0)*m1+torch.clamp(img[:, 0, :, :]*360 -
        120, 0.0, 60.0)*m2+torch.clamp(img[:, 0, :, :]*360-180, 0.0, 120.0)*m3+torch.clamp(img[:, 0, :, :]*360-300, 0.0, 60.0)*m4
    del m1, m2, m3, m4

    img = torch.stack((r, g, b), 1)
    del r, g, b
    #img[(img != img).detach()] = 0 # This causes memory error

    img = img.contiguous()
    img = torch.clamp(img, 0.0, 1.0)

    return img



def rgb_to_hsv(img):
    """Converts an RGB image to HSV
    PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

    :param img: RGB image
    :returns: HSV image
    :rtype: Tensor

    """
    img=torch.clamp(img, 10**(-9), 1.0)       

    # Shape (b, c, x, y)
    img = img.contiguous()

    # Shape (b, x, y)
    mx = torch.max(img, 1)[0]
    mn = torch.min(img, 1)[0]

    ones = Variable(torch.ones(mn.shape, device=torch.device('cuda'), dtype=torch.float))
    zero = Variable(torch.zeros(mn.shape, device=torch.device('cuda'), dtype=torch.float))
    df = torch.add(mx, torch.mul(ones*-1, mn))

    # Each channel is shape (b, x, y) tensor
    r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

    # New channel 0, hue (see: https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
    df_inv = ImageProcessing.non_nan_inv(df)        
    img[:, 0, :, :] = torch.where(df == zero, 
                                  zero,
                                  ((g-b)*df_inv)*r.eq(mx).float() + (2.0+(b-r)*df_inv)
                                  * g.eq(mx).float() + (4.0+(r-g)*df_inv)*b.eq(mx).float())
    img[:, 0, :, :] = img[:, 0, :, :]*60.0

    # Convert hue to range 0 to 360
    img[:, 0, :, :] = img[:, 0, :, :].lt(zero).float(
    )*(img[:, 0, :, :]+360) + img[:, 0, :, :].ge(zero).float()*(img[:, 0, :, :])

    img[:, 0, :, :] = img[:, 0, :, :]/360

    # Set saturation and value, remaining channels
    mx_inv = ImageProcessing.non_nan_inv(mx)
    img[:, 1, :, :] = torch.where(mx == zero,
                                  zero,
                                  mx.ne(zero).float()*(df*mx_inv) + mx.eq(zero).float()*(zero))
    img[:, 2, :, :] = mx

    img = torch.clamp(img, 10**(-9), 1.0)

    return img



def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out):
    """Applies a peicewise linear curve defined by a set of knot points to
    an image channel

    :param img: image to be adjusted
    :param C: predicted knot points of curve
    :returns: adjusted image
    :rtype: Tensor

    """
    curve_steps = C.shape[1]-1
    #curve_steps = C.shape[1]

    '''
    Compute the slope of the line segments
    '''
    slope = C[:, 1:]-C[:, 0:-1]

    '''
    Compute the squared difference between slopes
    '''
    slope_sqr_diff += ((slope[:, 1:]-slope[:, 0:-1])**2).sum(1)

    '''
    Use predicted line segments to compute scaling factors for the channel
    '''
    steps = torch.arange(0, slope.shape[1]-1, device=torch.device('cuda'))
    image_channel = torch.unsqueeze(img[:, channel_in, :, :], 1)  # expand dims to broadcast
    scale = C[:, 0].reshape(-1, 1, 1) + (slope[:, :-1].reshape(slope.shape[0], slope.shape[1] - 1, 1, 1) * 
                       (curve_steps * image_channel - steps.reshape(1, steps.shape[0], 1, 1))).sum(1)  # eq. 1

    img_copy = img.clone()
    img_copy[:, channel_out, :, :] = img[:, channel_out, :, :]*scale
    img_copy = torch.clamp(img_copy, 0.0, 1.0)

    return img_copy, slope_sqr_diff


def adjust_hsv(img, S):
    """Adjust the HSV channels of a HSV image using learnt curves

    :param img: image to be adjusted 
    :param S: predicted parameters of piecewise linear curves
    :returns: adjust image, regularisation term
    :rtype: Tensor, float

    """
    img = img.contiguous()
    batch_dim = img.shape[0]

    S1, S2, S3, S4 = torch.chunk(S, 4, dim=1)
    S1, S2, S3, S4 = torch.exp(S1), torch.exp(S2), torch.exp(S3), torch.exp(S4)

    slope_sqr_diff = Variable(torch.zeros(batch_dim, device=torch.device('cuda'))*0.0)

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

    img = img.contiguous()

    return img, slope_sqr_diff


def adjust_rgb(img, R):
    """Adjust the RGB channels of a RGB image using learnt curves

    :param img: image to be adjusted 
    :param S: predicted parameters of piecewise linear curves
    :returns: adjust image, regularisation term
    :rtype: Tensor, float

    """
    img = img.contiguous()
    batch_dim = img.shape[0]

    '''
    Extract the parameters of the three curves
    '''
    R1, R2, R3 = torch.chunk(R, 3, dim=1)
    R1, R2, R3 = torch.exp(R1), torch.exp(R2), torch.exp(R3)

    '''
    Apply the curve to the R channel 
    '''
    slope_sqr_diff = Variable(torch.zeros(batch_dim, device=torch.device('cuda'))*0.0)

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

    img = img.contiguous()

    return img, slope_sqr_diff


def adjust_lab(img, L):
    """Adjusts the image in LAB space using the predicted curves

    :param img: Image tensor
    :param L: Predicited curve parameters for LAB channels
    :returns: adjust image, and regularisation parameter
    :rtype: Tensor, float

    """

    img = img.contiguous()
    batch_dim = img.shape[0]

    '''
    Extract predicted parameters for each L,a,b curve
    '''
    L1, L2, L3 = torch.chunk(L, 3, dim=1)
    L1, L2, L3 = torch.exp(L1), torch.exp(L2), torch.exp(L3)

    slope_sqr_diff = Variable(torch.zeros(batch_dim, device=torch.device('cuda'))*0.0)

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

    img = img.contiguous()

    return img, slope_sqr_diff


def non_nan_inv(tensor):
    tensor_inv = torch.zeros(tensor.shape, device=torch.device('cuda'), dtype=torch.float)
    tensor_inv[tensor != 0] = 1/(tensor[tensor != 0])

    return tensor_inv
