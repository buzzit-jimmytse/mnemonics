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
""" Trainer for incremental learning. """
import sys
import copy
import argparse
import os
import torch
import torchvision
import warnings
import math
import utils.misc
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_resnetmtl_cifar as modified_resnetmtl_cifar
import models.modified_linear as modified_linear
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from utils.compute_features import compute_features
from utils.process_mnemonics import process_mnemonics
from utils.compute_accuracy import compute_accuracy
from trainer.incremental_train_and_eval import incremental_train_and_eval
from trainer.mnemonics_train import MnemonicsTrainer
try:
    import cPickle as pickle
except:
    import pickle
warnings.filterwarnings('ignore')

class Trainer(object):
    def __init__(self, the_args):
        """The class that contains the code for the train phase."""

        # Share argparse in this class
        self.args = the_args

        # Set the log folder name
        self.log_dir = './logs/'
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.save_path = self.log_dir + self.args.dataset + '_nfg' + str(self.args.nb_cl_fg) + '_ncls' + str(self.args.nb_cl) + '_nproto' + str(self.args.nb_protos) 
        if self.args.use_mtl:
            self.save_path += '_mtl'
        if self.args.fix_budget:
            self.save_path += '_fixbudget'            
        self.save_path += '_' + str(self.args.ckpt_label)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        # Set CUDA device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set the mnemonics trainer
        self.MnemonicsTrainer = MnemonicsTrainer(self.args)

        # Set dataset and transforms for CIFAR100
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))])
        self.transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))])
        self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform_train)
        self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform_test)
        self.evalset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=self.transform_test)

        # Set the backbone network
        self.network = modified_resnet_cifar.resnet32
        self.network_mtl = modified_resnetmtl_cifar.resnetmtl32

        # Set learning rate decay
        self.lr_strat = [int(self.args.epochs*0.5), int(self.args.epochs*0.75)]

        # Set dictionary size
        self.dictionary_size = self.args.dictionary_size

    def train(self):
        """The function for the train phase"""

        # Set tensorboard
        self.train_writer = SummaryWriter(logdir=self.save_path)

        # Load dictionary size
        dictionary_size = self.dictionary_size

        # Set array to record the accuracies
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 4, self.args.nb_runs))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 4, self.args.nb_runs))

        # Load the samples for CIFAR-100
        X_train_total = np.array(self.trainset.train_data)
        Y_train_total = np.array(self.trainset.train_labels)
        X_valid_total = np.array(self.testset.test_data)
        Y_valid_total = np.array(self.testset.test_labels)

        # Generate order list using random seed 1993. This operation follows the related papers e.g., iCaRL 
        for iteration_total in range(self.args.nb_runs):
            order_name = osp.join(self.save_path, "seed_{}_{}_order_run_{}.pkl".format(self.args.random_seed, self.args.dataset, iteration_total))
            print("Order name:{}".format(order_name))

            if osp.exists(order_name):
                print("Loading orders")
                order = utils.misc.unpickle(order_name)
            else:
                print("Generating orders")
                order = np.arange(self.args.num_classes)
                np.random.shuffle(order)
                utils.misc.savepickle(order, order_name)
            order_list = list(order)
            print(order_list)

        # Set empty lists and arrays
        X_valid_cumuls    = []
        X_protoset_cumuls = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []
        Y_train_cumuls    = []
        alpha_dr_herding  = np.zeros((int(self.args.num_classes/self.args.nb_cl),dictionary_size,self.args.nb_cl),np.float32)

        # Set the initial exemplars
        prototypes = np.zeros((self.args.num_classes,dictionary_size,X_train_total.shape[1],X_train_total.shape[2],X_train_total.shape[3]))
        for orde in range(self.args.num_classes):
            prototypes[orde,:,:,:,:] = X_train_total[np.where(Y_train_total==order[orde])]

        # Set the start iteration
        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1

        # Begin training
        for iteration in range(start_iter, int(self.args.num_classes/self.args.nb_cl)):
            # Initial model
            if iteration == start_iter:
                last_iter = 0
                tg_model = self.network(num_classes=self.args.nb_cl_fg)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("in_features:", in_features, "out_features:", out_features)
                ref_model = None
            elif iteration == start_iter+1:
                last_iter = iteration
                ref_model = copy.deepcopy(tg_model)
                tg_model = self.network_mtl(num_classes=self.args.nb_cl_fg)
                ref_dict = ref_model.state_dict()
                tg_dict = tg_model.state_dict()
                tg_dict.update(ref_dict)
                tg_model.load_state_dict(tg_dict)
                tg_model.to(self.device)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("in_features:", in_features, "out_features:", out_features)
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self.args.nb_cl)
                new_fc.fc1.weight.data = tg_model.fc.weight.data
                new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc
                lamda_mult = out_features*1.0 / self.args.nb_cl
            else:
                last_iter = iteration
                ref_model = copy.deepcopy(tg_model)
                in_features = tg_model.fc.in_features
                out_features1 = tg_model.fc.fc1.out_features
                out_features2 = tg_model.fc.fc2.out_features
                print("in_features:", in_features, "out_features1:", out_features1, "out_features2:", out_features2)
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, self.args.nb_cl)
                new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
                new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
                new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc
                lamda_mult = (out_features1+out_features2)*1.0 / (self.args.nb_cl)

            # Set lamda
            if iteration > start_iter:
                cur_lamda = self.args.lamda * math.sqrt(lamda_mult)
            else:
                cur_lamda = self.args.lamda
            if iteration > start_iter:
                print("Lamda for less forget is set to ", cur_lamda)

            # Add the current exemplars to the training set
            actual_cl = order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)]
            indices_train_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_train_total])
            indices_test_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_valid_total])

            X_train = X_train_total[indices_train_10]
            X_valid = X_valid_total[indices_test_10]
            X_valid_cumuls.append(X_valid)
            X_train_cumuls.append(X_train)
            X_valid_cumul = np.concatenate(X_valid_cumuls)
            X_train_cumul = np.concatenate(X_train_cumuls)

            Y_train = Y_train_total[indices_train_10]
            Y_valid = Y_valid_total[indices_test_10]
            Y_valid_cumuls.append(Y_valid)
            Y_train_cumuls.append(Y_train)
            Y_valid_cumul = np.concatenate(Y_valid_cumuls)
            Y_train_cumul = np.concatenate(Y_train_cumuls)

            if iteration == start_iter:
                X_valid_ori = X_valid
                Y_valid_ori = Y_valid
            else:
                X_protoset = np.concatenate(X_protoset_cumuls)
                Y_protoset = np.concatenate(Y_protoset_cumuls)

                if self.args.rs_ratio > 0:
                    scale_factor = (len(X_train) * self.args.rs_ratio) / (len(X_protoset) * (1 - self.args.rs_ratio))
                    rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset))*scale_factor))
                    rs_num_samples = int(len(X_train) / (1 - self.args.rs_ratio))
                    print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))
                X_train = np.concatenate((X_train,X_protoset),axis=0)
                Y_train = np.concatenate((Y_train,Y_protoset))

            # Launch the training loop
            print('Batch of classes number {0} arrives'.format(iteration+1))
            map_Y_train = np.array([order_list.index(i) for i in Y_train])
            map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

            # Imprint weights
            if iteration > start_iter and self.args.imprint_weights:
                print("Imprint weights")
                # Compute the average norm of old embdding
                old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
                average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
                tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
                num_features = tg_model.fc.in_features
                novel_embedding = torch.zeros((self.args.nb_cl, num_features))
                for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                    cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                    assert(len(np.where(cls_indices==1)[0])==dictionary_size)
                    self.evalset.test_data = X_train[cls_indices].astype('uint8')
                    self.evalset.test_labels = np.zeros(self.evalset.test_data.shape[0]) #zero labels
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=self.args.num_workers)
                    num_samples = self.evalset.test_data.shape[0]
                    cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                    cls_embedding = torch.mean(norm_features, dim=0)
                    novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
                tg_model.to(self.device)
                tg_model.fc.fc2.weight.data = novel_embedding.to(self.device)

            self.trainset.train_data = X_train.astype('uint8')
            self.trainset.train_labels = map_Y_train
            if iteration > start_iter and self.args.rs_ratio > 0 and scale_factor > 1:
                print("Weights from sampling:", rs_sample_weights)
                index1 = np.where(rs_sample_weights>1)[0]
                index2 = np.where(map_Y_train<iteration*self.args.nb_cl)[0]
                assert((index1==index2).all())
                train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
                trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size, \
                    shuffle=False, sampler=train_sampler, num_workers=self.args.num_workers)            
            else:
                trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size,
                    shuffle=True, num_workers=self.args.num_workers)
            self.testset.test_data = X_valid_cumul.astype('uint8')
            self.testset.test_labels = map_Y_valid_cumul
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size,
                shuffle=False, num_workers=self.args.num_workers)
            print('Max and min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
            print('Max and min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))

            # Set checkpoint name
            ckp_name = osp.join(self.save_path, 'run_{}_iteration_{}_model.pth'.format(iteration_total, iteration))
            print('Checkpoint name:', ckp_name)

            # Resume the saved models or set the new models
            if iteration==start_iter and self.args.resume_fg:
                print("Loading first group models from checkpoint")
                tg_model = torch.load(self.args.ckpt_dir_fg)
            elif self.args.resume and os.path.exists(ckp_name):
                print("Loading models from checkpoint")
                tg_model = torch.load(ckp_name)
            else:
                if iteration > start_iter:
                    ref_model = ref_model.to(self.device)

                    ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                        tg_model.parameters())

                    base_params = filter(lambda p: p.requires_grad,base_params)
                    tg_params_new =[{'params': base_params, 'lr': self.args.base_lr2, 'weight_decay': self.args.custom_weight_decay}, {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

                    tg_model = tg_model.to(self.device)
                    tg_optimizer = optim.SGD(tg_params_new, lr=self.args.base_lr2, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                else:
                    tg_params = tg_model.parameters()
                    tg_model = tg_model.to(self.device)
                    tg_optimizer = optim.SGD(tg_params, lr=self.args.base_lr1, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                if iteration > start_iter:
                    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=self.lr_strat, gamma=self.args.lr_factor)
                else:
                    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=self.lr_strat, gamma=self.args.lr_factor)           
                print("Incremental train")
                tg_model = incremental_train_and_eval(self.args.epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, cur_lamda, self.args.dist, self.args.K, self.args.lw_mr)   
                torch.save(tg_model, ckp_name)

            # Process the exemplars
            if self.args.fix_budget:
                nb_protos_cl = int(np.ceil(self.args.nb_protos*100./self.args.nb_cl/(iteration+1)))
            else:
                nb_protos_cl = self.args.nb_protos
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            num_features = tg_model.fc.in_features

            # Using herding startegy
            print('Updating exemplars')
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Possible exemplars in the feature space and projected on the L2 sphere
                self.evalset.test_data = prototypes[iter_dico].astype('uint8')
                self.evalset.test_labels = np.zeros(self.evalset.test_data.shape[0]) #zero labels
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers)
                num_samples = self.evalset.test_data.shape[0]            
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                # Ranking of the potential exemplars
                mu  = np.mean(D,axis=1)
                index1 = int(iter_dico/self.args.nb_cl)
                index2 = iter_dico % self.args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t,D)
                    ind_max = np.argmax(tmp_t)

                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t+mu-D[:,ind_max]

            # Prepare the protoset
            X_protoset_cumuls = []
            Y_protoset_cumuls = []

            # Calculate the mean of exemplars
            print('Computing the mean of exemplars')
            class_means = np.zeros((64,100,2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]

                    # Collect data in the feature space for each class
                    self.evalset.test_data = prototypes[iteration2*self.args.nb_cl+iter_dico].astype('uint8')
                    self.evalset.test_labels = np.zeros(self.evalset.test_data.shape[0]) #zero labels
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    num_samples = self.evalset.test_data.shape[0]
                    mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)

                    self.evalset.test_data = prototypes[iteration2*self.args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    D2 = mapped_prototypes2.T
                    D2 = D2/np.linalg.norm(D2,axis=0)

                    # iCaRL
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico,np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))

                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

                    # Nearest neighbor upper bound
                    alph = np.ones(dictionary_size)/dictionary_size
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

            torch.save(class_means, osp.join(self.save_path, 'run_{}_iteration_{}_class_means.pth'.format(iteration_total, iteration)))

            # Mnemonics
            print("Mnemonics training")
            # Initialize the mnemonics
            self.mnemonics, self.mnemonics_lrs, self.mnemonics_label = self.MnemonicsTrainer.mnemonics_init_with_images_cifar(iteration, start_iter, self.args.nb_cl_fg, self.args.nb_cl, X_protoset_cumuls, Y_protoset_cumuls, order_list, self.device)
            # Train the mnemonics
            self.mnemonics, self.mnemonics_lrs, self.mnemonics_label = self.MnemonicsTrainer.mnemonics_train(tg_model, trainloader, testloader, iteration, start_iter, self.device)
            # Process the mnemonics and set them as the exemplars
            X_protoset_cumuls, Y_protoset_cumuls = process_mnemonics(X_protoset_cumuls, Y_protoset_cumuls, self.mnemonics, self.mnemonics_label, order_list)        

            # Get current class means
            current_means = class_means[:, order[range(0,(iteration+1)*self.args.nb_cl)]]

            # Set iteration labels
            if iteration == start_iter:
                is_start_iteration = True
            else:
                is_start_iteration = False

            # Calculate validation error of model on the first nb_cl classes
            map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
            print('Computing accuracy on the original batch of classes')
            self.evalset.test_data = X_valid_ori.astype('uint8')
            self.evalset.test_labels = map_Y_valid_ori
            evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=self.args.num_workers)
            ori_acc, fast_fc = compute_accuracy(tg_model, tg_feature_model, current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=is_start_iteration, maml_lr=self.args.maml_lr, maml_epoch=self.args.maml_epoch)
            top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
            self.train_writer.add_scalar('ori_acc/cosine', float(ori_acc[0]), iteration)
            self.train_writer.add_scalar('ori_acc/nearest_neighbor', float(ori_acc[1]), iteration)
            self.train_writer.add_scalar('ori_acc/fc_finetune', float(ori_acc[3]), iteration)

            # Calculate validation error of model on the cumul of classes
            map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
            print('Computing cumulative accuracy')
            self.evalset.test_data = X_valid_cumul.astype('uint8')
            self.evalset.test_labels = map_Y_valid_cumul
            evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=self.args.num_workers)        
            cumul_acc, _ = compute_accuracy(tg_model, tg_feature_model, current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=is_start_iteration, fast_fc=fast_fc, maml_lr=self.args.maml_lr, maml_epoch=self.args.maml_epoch)
            top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
            self.train_writer.add_scalar('cumul_acc/cosine', float(cumul_acc[0]), iteration)
            self.train_writer.add_scalar('cumul_acc/nearest_neighbor', float(cumul_acc[1]), iteration)
            self.train_writer.add_scalar('cumul_acc/fc_finetune', float(cumul_acc[3]), iteration)

        # Save data and close tensorboard
        torch.save(top1_acc_list_ori, osp.join(self.save_path, 'run_{}_top1_acc_list_ori.pth'.format(iteration_total)))
        torch.save(top1_acc_list_cumul, osp.join(self.save_path, 'run_{}_top1_acc_list_cumul.pth'.format(iteration_total)))
        self.train_writer.close()