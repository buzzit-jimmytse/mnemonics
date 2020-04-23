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
"""Calculate the features."""
import torch
import numpy as np
from torchvision import models
from utils.misc import *

def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    """The function for computing features.
    Args:
      tg_feature_model: the backbone model without fc classifier.
      evalloader: test dataloader.
      num_samples: the number of samples.
      num_features: feature dim.
      device: the GPU device index.
    Returns:
      the target feature.
    """

    # Set CUDA device if it is necessary
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluation mode for the models
    tg_feature_model.eval()

    # Compute features
    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(tg_feature_model(inputs))
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features
