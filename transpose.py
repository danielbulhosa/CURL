import numpy as np


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