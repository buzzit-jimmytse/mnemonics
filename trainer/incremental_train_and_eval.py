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
"""Incremental learning and evaluation function."""
import torch
import tqdm
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *

# Set empty list for features and secores
cur_features = []
ref_features = []
old_scores = []
new_scores = []

def get_ref_features(self, inputs, outputs):
    """The function for getting reference model features."""
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    """The function for getting current model features."""
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    """The function for getting old scores."""
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    """The function for getting new scores."""
    global new_scores
    new_scores = outputs

def incremental_train_and_eval(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iteration, lamda, dist, K, lw_mr, fix_bn=False, weight_per_class=None, device=None):
    """Incremental learning and evaluation function.
    Args:
      epochs: the number of total epochs for this incremental phase.
      tg_model: the current backbone model.
      ref_model: the old backbone model.
      tg_optimizer: optimizer for the backbone model and fc classifier.
      tg_lr_scheduler: learning rate scheduler for the backbone model and fc classifier.
      trainloader: train dataloader.
      testloader: test dataloader.
      iteration: the current incremental phase index.
      start_iteration: the initial incremental phase index.
      lamda, dist, K, lw_mr: some hyperparameters for training, see details in LUCIR.
      fix_bn, weight_per_class: not applied in the current released version. 
      device: CUDA device index.

    Return:
      The trained model.
    """

    # Set CUDA device if it is empty
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Register forward hook for the following incremental phases.
    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    # Start training
    for epoch in range(epochs):
        # Set the model to train mode
        tg_model.train()

        # Fix batch norm if it is necessary
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set the initial values for different losses and numbers of correct samples
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        correct = 0
        total = 0

        # Process learning rate scheduler
        tg_lr_scheduler.step()

        # Print information for the current epoch
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())

        # Using tqdm
        tqdm_gen = tqdm.tqdm(trainloader)

        # Starting training for current incremental phase
        for batch_idx, (inputs, targets) in enumerate(tqdm_gen):

            # Send data to GPU device
            inputs, targets = inputs.to(device), targets.to(device)

            # Set the gradient for the optimizer to zero
            tg_optimizer.zero_grad()

            # Feature extraction
            outputs = tg_model(inputs)

            # Calculate the losses
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)
                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(device)) * lamda
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                outputs_bs = torch.cat((old_scores, new_scores), dim=1)
                assert(outputs_bs.size()==outputs.size())
                # Get groud truth scores
                gt_index = torch.zeros(outputs_bs.size()).to(device)
                gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
                gt_scores = outputs_bs.masked_select(gt_index)
                # Get top-K scores on novel classes
                max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
                # The index of hard samples, i.e., samples of old classes
                hard_index = targets.lt(num_old_classes)
                hard_num = torch.nonzero(hard_index).size(0)
                if  hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                    max_novel_scores = max_novel_scores[hard_index]
                    assert(gt_scores.size() == max_novel_scores.size())
                    assert(gt_scores.size(0) == hard_num)
                    loss3 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1), torch.ones(hard_num*K).to(device)) * lw_mr
                else:
                    loss3 = torch.zeros(1).to(device)
                loss = loss1 + loss2 + loss3

            # Update
            loss.backward()
            tg_optimizer.step()

            # Record losses and predictions
            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print losses and accuracies
        if iteration == start_iteration:
            print('Train set: {}, train Loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            print('Train set: {}, train Loss1: {:.4f}, train Loss2: {:.4f}, train Loss3: {:.4f}, train Loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1), train_loss/(batch_idx+1), 100.*correct/total))

        # Run evaluation for the current incremental phase

        # The following steps are similar to training
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test Loss: {:.4f} accuracy: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    # Remove refuster forward hook if it is necessary
    if iteration > start_iteration:
        print("Removing register forward hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    return tg_model
