import torch
import torch.nn as nn

def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out):
    """Applies a peicewise linear curve defined by a set of knot points to
    an image channel

    :param img: image to be adjusted
    :param C: predicted knot points of curve
    :returns: adjusted image
    :rtype: Tensor

    """
    curve_steps = C.shape[1]-1

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
    steps = nn.Parameter(torch.arange(0, slope.shape[1]-1), requires_grad=False)
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

    slope_sqr_diff = None

    '''
    Adjust Hue channel based on Hue using the predicted curve
    '''
    img_copy, slope_sqr_diff = apply_curve(
        img, S1, slope_sqr_diff, channel_in=0, channel_out=0)

    '''
    Adjust Saturation channel based on Hue using the predicted curve
    '''
    img_copy, slope_sqr_diff = apply_curve(
        img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1)

    '''
    Adjust Saturation channel based on Saturation using the predicted curve
    '''
    img_copy, slope_sqr_diff = apply_curve(
        img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1)

    '''
    Adjust Value channel based on Value using the predicted curve
    '''
    img_copy, slope_sqr_diff = apply_curve(
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
    slope_sqr_diff = None

    img_copy, slope_sqr_diff = apply_curve(
        img, R1, slope_sqr_diff, channel_in=0, channel_out=0)

    '''
    Apply the curve to the G channel 
    '''
    img_copy, slope_sqr_diff = apply_curve(
        img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1)

    '''
    Apply the curve to the B channel 
    '''
    img_copy, slope_sqr_diff = apply_curve(
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

    slope_sqr_diff = None

    '''
    Apply the curve to the L channel 
    '''
    img_copy, slope_sqr_diff = apply_curve(
        img, L1, slope_sqr_diff, channel_in=0, channel_out=0)

    '''
    Now do the same for the a channel
    '''
    img_copy, slope_sqr_diff = apply_curve(
        img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1)

    '''
    Now do the same for the b channel
    '''
    img_copy, slope_sqr_diff = apply_curve(
        img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2)

    img = img_copy.clone()
    del img_copy

    img = img.contiguous()

    return img, slope_sqr_diff