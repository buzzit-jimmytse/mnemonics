##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## MPI for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Accuracy calculation function."""
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from scipy.spatial.distance import cdist
from utils.misc import *

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

def compute_accuracy(tg_model, tg_feature_model, class_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=False, fast_fc=None, scale=None, print_info=True, device=None, maml_lr=0.1, maml_epoch=50):
    """The function for computing accuracies.
    Args:
      tg_model: the backbone model with fc classifier.
      tg_feature_model: the backbone model without fc classifier.
      class_means: mean values for classes
      X_protoset_cumuls: exemplar set.
      Y_protoset_cumuls: labels for the emexplar set.
      evalloader: test dataloader.
      order_list: the list of the true class order and relative class order.
      is_start_iteration: indicator for the first incremental phase.
      scale: scale or not.
      print_info: print the accuracy or not.
      maml_lr: learning rate for fc classifier finetune.
      maml_epoch: the number of epochs for fc classifier finetune.
      device: the GPU device index.
    Returns:
      [cnn_acc, icarl_acc, ncm_acc, maml_acc]: test accuracies.
      fast_fc: finetuned fc classifier.
    """
    
    # Set CUDA device if it is necessary
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluation mode for the models
    tg_model.eval()
    tg_feature_model.eval()

    # Finetune fc classifier if it hasn't been calculated
    if fast_fc is None:
        transform_proto = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
        protoset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_proto)
        X_protoset_array = np.array(X_protoset_cumuls).astype('uint8')
        protoset.test_data = X_protoset_array.reshape(-1, X_protoset_array.shape[2], X_protoset_array.shape[3], X_protoset_array.shape[4])
        Y_protoset_cumuls = np.array(Y_protoset_cumuls).reshape(-1)
        map_Y_protoset_cumuls = map_labels(order_list, Y_protoset_cumuls)
        protoset.test_labels = map_Y_protoset_cumuls
        protoloader = torch.utils.data.DataLoader(protoset, batch_size=128, shuffle=True, num_workers=2)  

        fast_fc = torch.from_numpy(np.float32(class_means[:,:,0].T)).to(device)
        fast_fc.requires_grad=True

        epoch_num = maml_epoch
        for epoch_idx in range(epoch_num):
            for the_inputs, the_targets in protoloader: 
                the_inputs, the_targets = the_inputs.to(device), the_targets.to(device)
                the_features = tg_feature_model(the_inputs)
                the_logits = F.linear(F.normalize(torch.squeeze(the_features), p=2,dim=1), F.normalize(fast_fc, p=2, dim=1))
                the_loss = F.cross_entropy(the_logits, the_targets)
                the_grad = torch.autograd.grad(the_loss, fast_fc)
                fast_fc = fast_fc - maml_lr * the_grad[0]

    # Set the number of correct samples to zero
    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    correct_maml = 0
    total = 0

    # Computing the accuracies
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            # Compute score for cosine classifier
            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_feature = np.squeeze(tg_feature_model(inputs))
            # Compute score for nearest neighbor classifier
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            # Compute score for the upper bound of nearest neighbor classifier
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()

            # Compute score for finetuned fc classifier
            the_logits = F.linear(F.normalize(torch.squeeze(outputs_feature), p=2,dim=1), F.normalize(fast_fc, p=2, dim=1))
            _, predicted_maml = the_logits.max(1)
            correct_maml += predicted_maml.eq(targets).sum().item()

    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total
    maml_acc = 100.*correct_maml/total

    # Print test accuracies
    if print_info:
        print("  Accuracy for cosine classifier               :\t\t{:.2f} %".format(100.*cnn_acc))
        print("  Accuracy for nearest neighbor classifier     :\t\t{:.2f} %".format(100.*icarl_acc))
        print("  Accuracy for finetuned fc classifier         :\t\t{:.2f} %".format(100.*maml_acc))

    return [cnn_acc, icarl_acc, ncm_acc, maml_acc], fast_fc