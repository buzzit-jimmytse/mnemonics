##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## MPI for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repository. """
import os
import argparse
import numpy as np
from trainer.train import Trainer
from utils.gpu_tools import occupy_memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0') # GPU id 
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet'])
    parser.add_argument('--data_dir', default='data/seed_1993_subset_100_imagenet/data', type=str)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--nb_cl_fg', default=50, type=int)
    parser.add_argument('--nb_cl', default=10, type=int)
    parser.add_argument('--nb_protos', default=20, type=int)
    parser.add_argument('--nb_runs', default=1, type=int)
    parser.add_argument('--epochs', default=160, type=int)
    parser.add_argument('--T', default=2, type=float)
    parser.add_argument('--beta', default=0.25, type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_fg', action='store_true')
    parser.add_argument('--ckpt_dir_fg', type=str, default='-')
    parser.add_argument('--fix_budget', action='store_true')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--ckpt_label', type=str, default='exp01')
    parser.add_argument('--use_mtl', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--load_iter', default=0, type=int)
    parser.add_argument('--dictionary_size', default=500, type=int)
    parser.add_argument('--mimic_score', action='store_true')
    parser.add_argument('--lw_ms', default=1, type=float)
    parser.add_argument('--rs_ratio', default=0, type=float)
    parser.add_argument('--imprint_weights', action='store_true')
    parser.add_argument('--less_forget', action='store_true')
    parser.add_argument('--lamda', default=5, type=float)
    parser.add_argument('--adapt_lamda', action='store_true')
    parser.add_argument('--dist', default=0.5, type=float)
    parser.add_argument('--K', default=2, type=int)
    parser.add_argument('--lw_mr', default=1, type=float)
    parser.add_argument('--random_seed', default=1993, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=100, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--base_lr1', default=0.1, type=float)
    parser.add_argument('--base_lr2', default=0.1, type=float)
    parser.add_argument('--lr_factor', default=0.1, type=float)
    parser.add_argument('--custom_weight_decay', default=5e-4, type=float)
    parser.add_argument('--custom_momentum', default=0.9, type=float)
    parser.add_argument('--load_ckpt_prefix', type=str, default='-')
    parser.add_argument('--load_order', type=str, default='-')
    parser.add_argument('--maml_lr', default=0.1, type=float)
    parser.add_argument('--maml_epoch', default=50, type=int)
    parser.add_argument('--mnemonics_images_per_class_per_step', default=1, type=int)    
    parser.add_argument('--mnemonics_steps', default=20, type=int)    
    parser.add_argument('--mnemonics_epochs', default=5, type=int)    
    parser.add_argument('--mnemonics_lr', type=float, default=0.01)
    parser.add_argument('--mnemonics_decay_factor', type=float, default=0.5)
    parser.add_argument('--mnemonics_outer_lr', type=float, default=1e-6)
    parser.add_argument('--mnemonics_total_epochs', type=int, default=10)
    parser.add_argument('--mnemonics_decay_epochs', type=int, default=40)

    the_args = parser.parse_args()

    assert(the_args.nb_cl_fg % the_args.nb_cl == 0)
    assert(the_args.nb_cl_fg >= the_args.nb_cl)

    print(the_args)

    np.random.seed(the_args.random_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
    print('Using gpu:', the_args.gpu)

    occupy_memory(the_args.gpu)
    print('Occupy GPU memory in advance')

    if the_args.phase=='train':
        trainer = Trainer(the_args)
        trainer.train()
    elif the_args.phase=='eval':
        trainer = Trainer(the_args)
        trainer.eval()
    else:
        raise ValueError('Please set correct phase')




