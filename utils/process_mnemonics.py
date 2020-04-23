##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## MPI for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.optim as optim
import torchvision
import time
import os
import argparse
import numpy as np
"""Functions for process mnemonics exemplars."""

def tensor2im(input_image, imtype=np.uint8):
    """"Transfer images in tensors to numpy arrays
    Args:
      input_image: the image to transfer.
      imtype: data type of outputs.
    Returns:
      Images in numpy arrays.
    """
    # Mean and std for dataloader
    mean = [0.5071,  0.4866,  0.4409]
    std = [0.2009,  0.1984,  0.2023]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        # Convert it into a numpy array
        image_numpy = image_tensor.cpu().detach().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # Recover the mean and std
        for i in range(len(mean)): 
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        # Transfer from [0,1] to [0,255]
        image_numpy = image_numpy * 255
        # From (channels, height, width) to (height, width, channels)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def map_labels_back(order_list, Y_set):
    """The function for mapping labels between the true class order and relative class order (reversed version).
    Args:
      order_list: the list of the true class order and relative class order.
      Y_set: the labels before mapping.
    Return:
      The labels after mapping.
    """
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def process_mnemonics(X_protoset_cumuls, Y_protoset_cumuls, mnemonics, mnemonics_label, order_list):
    """Process mnemonics exemplars
    Args:
       X_protoset_cumuls: exemplar set before mnemonics training.
       Y_protoset_cumuls: labels for the emexplar set before mnemonics training.
       mnemonics: the updated mnemonics exemplars.
       mnemonics_label: the labels mnemonics exemplars.
       order_list: the list of the true class order and relative class order.
    Returns:
       X_protoset_cumuls: exemplar set after mnemonics training.
       Y_protoset_cumuls: labels for the emexplar set after mnemonics training.
    """
    mnemonics_list = []
    mnemonics_label_list = []
    for idx in range(len(mnemonics)):
        this_mnemonics = []
        for sub_idx in range(len(mnemonics[idx])):
            processed_img = tensor2im(mnemonics[idx][sub_idx]) 
            this_mnemonics.append(processed_img)
        this_mnemonics = np.array(this_mnemonics)
        mnemonics_list.append(this_mnemonics)
        mnemonics_label_list.append(mnemonics_label.cpu().detach().numpy())
    mnemonics_array = np.array(mnemonics_list)
    mnemonics_label_array = np.array(mnemonics_label_list)

    mnemonics_array = mnemonics_array.transpose(1,0,2,3,4)
    mnemonics_label_array = mnemonics_label_array.transpose(1,0)

    map_back_Y = []
    for idx in range(len(mnemonics_label_array)):
        this_new = []
        this_array = mnemonics_label_array[idx]
        for sub_idx in range(len(this_array)):
            this_new.append(order_list[int(this_array[sub_idx])])
        this_new = np.array(this_new)
        map_back_Y.append(this_new)

    diff = len(X_protoset_cumuls) - len(mnemonics_array)
    for idx in range(len(mnemonics_array)):
        X_protoset_cumuls[idx+diff] = mnemonics_array[idx]
        Y_protoset_cumuls[idx+diff] = map_back_Y[idx]

    return X_protoset_cumuls, Y_protoset_cumuls
