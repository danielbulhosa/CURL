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
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
import os
import util
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from PIL import Image 

np.set_printoptions(threshold=sys.maxsize)


def get_data_ids(img_ids_filepath):
    with open(img_ids_filepath) as f:
        '''
        Load the image ids into a list data structure
        '''
        image_ids = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        image_ids_list = [int(x.rstrip()) for x in image_ids if not x.startswith('.')]

    return image_ids_list


def get_data_dict(data_dirpath):
    data_dirs = sorted(os.listdir(data_dirpath))
    try: 
        input_dir = [directory for directory in data_dirs if 'input' in directory][0]
        output_dir = [directory for directory in data_dirs if 'output' in directory][0]
        logging.info("Using {} as input directory and {} as output directory.".format(input_dir, output_dir))
    except IndexError:
        raise OSError("{} must contain a directories containing the words 'input', 'output' respectively".format(data_dirpath))

    full_input_dir = data_dirpath + input_dir + os.path.sep
    full_output_dir = data_dirpath + output_dir + os.path.sep
        
    input_imgs = [filename for filename in sorted(os.listdir(full_input_dir)) if not filename.startswith('.')]
    output_imgs = [filename for filename in sorted(os.listdir(full_output_dir)) if not filename.startswith('.')]
    
    assert input_imgs == output_imgs, "Input and output image directories should have the same file names."
    
    idxs = [int(os.path.splitext(filename)[0]) for filename in input_imgs]
    data_dict = {}
    
    for idx, input_img, output_img in zip(idxs, input_imgs, output_imgs):
        data_dict[idx] = {'input_img': full_input_dir + input_img, 
                          'output_img': full_output_dir + output_img}
        
    return data_dict


def filter_data_dict(data_dict, image_id_list):
    filtered_dict = {}
    for new_idx, idx in enumerate(image_id_list):
         filtered_dict[new_idx] = data_dict[idx]
    
    return filtered_dict


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, transform=None, normaliser=2 ** 8 - 1, is_valid=False, is_inference=False):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param transform: PyTorch image transformations to apply to the images
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.data_dict = data_dict
        self.normaliser = normaliser
        self.is_valid = is_valid
        self.is_inference = is_inference
        
    @staticmethod
    def pad(image_tensor, crop_height, crop_width):
        image_height, image_width = image_tensor.shape[1:]

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            return TF.pad(image_tensor, padding_ltrb, fill=0)
        else:
            return image_tensor
        

    def __len__(self):
        """Returns the number of images in the dataset

        :returns: number of images in the dataset
        :rtype: Integer

        """
        return (len(self.data_dict.keys()))

    def __getitem__(self, idx):
        """Returns a pair of images with the given identifier. This is lazy loading
        of data into memory. Only those image pairs needed for the current batch
        are loaded.

        :param idx: image pair identifier
        :returns: dictionary containing input and output images and their identifier
        :rtype: dictionary

        """

        input_img = util.ImageProcessing.load_image(
            self.data_dict[idx]['input_img'], normaliser=self.normaliser)
        output_img = util.ImageProcessing.load_image(
            self.data_dict[idx]['output_img'], normaliser=self.normaliser)

        if self.normaliser==1:
            input_img, output_img = input_img.astype(np.uint8), output_img.astype(np.uint8)

        input_img, output_img = TF.to_tensor(input_img), TF.to_tensor(output_img)
        # FIXME - better options here:
        #   - Have a crop size range instead of a fixed number
        #   - Analyze data to come up with better fixed crop size
        #   - Do a random crop instead of a center crop
        crop_h, crop_w = 256, 256
        input_img, output_img = self.pad(input_img, crop_h, crop_w), self.pad(output_img, crop_h, crop_w)
        input_img, output_img = TF.center_crop(input_img, (crop_h, crop_w)), TF.center_crop(output_img, (crop_h, crop_w))

        if not self.is_valid and not self.is_inference:

                # Random horizontal flipping
                if random.random() > 0.5:
                    input_img, output_img = TF.hflip(input_img), TF.hflip(output_img)

                # Random vertical flipping
                if random.random() > 0.5:
                    input_img, output_img = TF.vflip(input_img), TF.vflip(output_img)
                  
                # Random modulo 90 degree rotation
                rotation = int(np.random.choice([-90, 0, 90, 180]))
                if rotation != 0:
                    input_img, output_img = TF.rotate(input_img, rotation, expand=True),\
                                            TF.rotate(output_img, rotation, expand=True)


        return {'input_img': input_img, 'output_img': output_img,
                'name': self.data_dict[idx]['input_img'].split("/")[-1]}
