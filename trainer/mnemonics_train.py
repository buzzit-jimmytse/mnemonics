##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## MPI for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for mnemonics. """
import torch
import tqdm
import time
import os
import copy
import argparse
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils.misc import *
from tensorboardX import SummaryWriter

def map_labels(order_list, Y_set):
    """The function for mapping labels between the true class order and relative class order.
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

class MnemonicsTrainer(object):
    """The class that contains trainer for mnemonics."""
    def __init__(self, the_args):
        # Load the argparse and set some basic parameters
        self.args = the_args
        self.T = self.args.mnemonics_steps * self.args.mnemonics_epochs
        self.img_size = 32

    def mnemonics_init_with_images_cifar(self, iteration, start_iter, num_classes, num_classes_incremental, X_protoset_cumuls, Y_protoset_cumuls, order_list, device):
        """The function for initializing mnemonics on CIFAR100.
        Args:
          iteration: the current incremental phase index.
          start_iter: the initial incremental phase index.
          number_classes: the number of the classes for first incremental phase.
          num_classes_incremental: the number of classes for the following incremental phases.
          X_protoset_cumuls: exemplar set before mnemonics training.
          Y_protoset_cumuls: labels for the emexplar set before mnemonics training.
          order_list: the list of the true class order and relative class order.
          device: the GPU device index.
        Returns:
          self.mnemonics: the updated mnemonics exemplars.
          self.mnemonics_lrs: learning rates for mnemonics exemplars.
          self.mnemonics_label: the labels mnemonics exemplars.
        """

        # Set the parameter list for mnemonics
        self.mnemonics = nn.ParameterList()
        self.mnemonics_lrs = nn.ParameterList()

        # Set transform for CIFAR100
        transform_proto = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
        
        # Initialize the learning rates for mnemonics
        raw_init_mnemonics_lr = torch.tensor(self.args.mnemonics_lr, device=device)
        raw_init_mnemonics_lr = raw_init_mnemonics_lr.repeat(self.T, 1)
        self.raw_mnemonics_lrs = raw_init_mnemonics_lr.expm1_().requires_grad_()
        self.mnemonics_lrs.append(nn.Parameter(self.raw_mnemonics_lrs))

        # Initialize the mnemonics exemplars for the first incremental phase
        if iteration == start_iter:
            num_per_step = num_classes * self.args.mnemonics_images_per_class_per_step
            X_protoset_array = np.array(X_protoset_cumuls).astype('uint8')
            Y_protoset_cumuls = np.array(Y_protoset_cumuls).reshape(-1)
            map_Y_protoset_cumuls = map_labels(order_list, Y_protoset_cumuls)
            
            for step_idx in range(self.args.mnemonics_steps):               
                mnemonics_data = torch.zeros(num_per_step, 3, self.img_size, self.img_size, device=device, requires_grad=True)
                for img_idx in range(self.args.mnemonics_images_per_class_per_step):
                    for cls_idx in range(num_classes):
                        the_img = X_protoset_array[cls_idx][img_idx]
                        the_PIL_image = Image.fromarray(the_img)
                        the_PIL_image = transform_proto(the_PIL_image)
                        mnemonics_data[img_idx*self.args.mnemonics_images_per_class_per_step+cls_idx]=the_PIL_image
                self.mnemonics.append(nn.Parameter(mnemonics_data))
            self.mnemonics_label = torch.arange(num_classes).repeat(self.args.mnemonics_images_per_class_per_step).to(device)
        # Initialize the mnemonics exemplars for the following incremental phases
        else:
            num_per_step = num_classes_incremental * self.args.mnemonics_images_per_class_per_step
            X_protoset_array = np.array(X_protoset_cumuls).astype('uint8')[len(X_protoset_cumuls)-1-num_classes_incremental:]
            Y_protoset_cumuls = np.array(Y_protoset_cumuls).reshape(-1)[len(Y_protoset_cumuls)-1-num_classes_incremental:]
            map_Y_protoset_cumuls = map_labels(order_list, Y_protoset_cumuls)
            
            for step_idx in range(self.args.mnemonics_steps):               
                mnemonics_data = torch.zeros(num_per_step, 3, self.img_size, self.img_size, device=device, requires_grad=True)
                for img_idx in range(self.args.mnemonics_images_per_class_per_step):
                    for cls_idx in range(num_classes_incremental):
                        the_img = X_protoset_array[cls_idx][img_idx]
                        the_PIL_image = Image.fromarray(the_img)
                        the_PIL_image = transform_proto(the_PIL_image)
                        mnemonics_data[img_idx*self.args.mnemonics_images_per_class_per_step+cls_idx]=the_PIL_image
                self.mnemonics.append(nn.Parameter(mnemonics_data))
            self.mnemonics_label = torch.arange(num_classes_incremental).repeat(self.args.mnemonics_images_per_class_per_step).to(device)
            self.mnemonics_label = self.mnemonics_label + num_classes + (iteration-start_iter-1)*num_classes_incremental
        return self.mnemonics, self.mnemonics_lrs, self.mnemonics_label   

    def mnemonics_train(self, tg_model, trainloader, testloader, iteration, start_iteration, device=None):
        """The function for training mnemonics.
        Args:
          tg_model: the backbone model.
          trainloader: train dataloader.
          testloader: test dataloader.
          iteration: the current incremental phase index.
          start_iteration: the initial incremental phase index.
          device: the GPU device index.
        Returns:
          self.mnemonics: the updated mnemonics exemplars.
          self.mnemonics_lrs: learning rates for mnemonics exemplars.
          self.mnemonics_label: the labels mnemonics exemplars.
        """
        # Set CUDA device if it is empty
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set backbone model as feature extractor and use evaluation mode
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        tg_feature_model.eval()

        # Set the optimizer adn learning rate scheduler for mnemonics exemplars
        self.mnemonics_optimizer = optim.Adam(self.mnemonics.parameters(), lr=self.args.mnemonics_outer_lr, betas=(0.5, 0.999))
        self.mnemonics_lr_scheduler = optim.lr_scheduler.StepLR(self.mnemonics_optimizer, step_size=self.args.mnemonics_decay_epochs, gamma=self.args.mnemonics_decay_factor)

        # Start training
        # In this simplified version, we use only one fc layer to train the mnemonics exemplars.
        # It makes this released repository can be run on computers with small GPU memory.
        # The original version, we update the whole model instead of one fc layer.
        for epoch in range(self.args.mnemonics_total_epochs):
            train_loss = 0
            correct = 0
            total = 0
            self.mnemonics_lr_scheduler.step()

            if torch.cuda.is_available():
                mnemonics_label = self.mnemonics_label.type(torch.cuda.LongTensor)
            else:
                mnemonics_label = self.mnemonics_label.type(torch.LongTensor)

            for batch_idx, (q_inputs, q_targets) in enumerate(trainloader):
                q_inputs, q_targets = q_inputs.to(device), q_targets.to(device)
                self.mnemonics_optimizer.zero_grad()

                if iteration == start_iteration:
                    fast_fc = copy.deepcopy(tg_model.fc.weight)
                else:
                    fast_fc = torch.cat((tg_model.fc.fc1.weight, tg_model.fc.fc2.weight), dim=0)
                q_outputs = tg_feature_model(q_inputs)
                for mnemonics_epochs_idx in range(self.args.mnemonics_epochs):
                    for mnemonics_steps_idx in range(self.args.mnemonics_steps):
                        mnemonics_outputs = tg_feature_model(self.mnemonics[mnemonics_steps_idx])
                        the_logits = F.linear(F.normalize(torch.squeeze(mnemonics_outputs), p=2,dim=1), F.normalize(fast_fc, p=2, dim=1))
                        the_loss = F.cross_entropy(the_logits, mnemonics_label)
                        the_grad = torch.autograd.grad(the_loss, fast_fc)
                        fast_fc = fast_fc - self.mnemonics_lrs[0][mnemonics_steps_idx] * the_grad[0]

                q_logits = F.linear(F.normalize(torch.squeeze(q_outputs), p=2,dim=1), F.normalize(fast_fc, p=2, dim=1))
                q_loss = F.cross_entropy(q_logits, q_targets)
                q_loss.backward()
                self.mnemonics_optimizer.step()

                train_loss += q_loss.item()
                _, predicted = q_logits.max(1)
                total += q_targets.size(0)
                correct += predicted.eq(q_targets).sum().item()

        return self.mnemonics, self.mnemonics_lrs, self.mnemonics_label