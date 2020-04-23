##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## MPI for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" GPU tools. """
import os
import torch
import time

def check_memory(cuda_device):
    """Check GPU memory.
    Arg:
      cuda_device: the GPU device index.
    Returns:
      total: total GPU memory
      used: used GPU memory
    """
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used

def occupy_memory(cuda_device):
    """Occupy GPU memory in advance.
    Arg:
      cuda_device: the GPU device index.
    """
    total, used = check_memory(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.90)
    print('Total memory: ' + str(total) + ', used memory: ' + str(used))
    block_mem = max_mem - used
    if block_mem > 0:
        x = torch.cuda.FloatTensor(256, 1024, block_mem)
        del x

def set_gpu(cuda_device):
    """Setting specific GPU.
    Arg:
      cuda_device: the GPU device index.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print('Using gpu:', cuda_device)