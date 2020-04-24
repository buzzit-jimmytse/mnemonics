## Mnemonics Training: Multi-Class Incremental Learning without Forgetting

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8)](https://pytorch.org/)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/yaoyao-liu/mnemonics)

This repository contains the PyTorch implementation for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "[Mnemonics Training: Multi-Class Incremental Learning without Forgetting](https://arxiv.org/pdf/2002.10211.pdf)" by [Yaoyao Liu](https://yyliu.net/), [Yuting Su](https://www.iti-tju.org/#/people/suyutingEnglish), [An-An Liu](https://www.iti-tju.org/#/people/liuananEnglish), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/), and [Qianru Sun](https://qianrusun1015.github.io).

This is a preliminary released version. Welcome to report issues and bugs for this repository. If you have any questions on this repository or the related paper, feel free to [create an issue](https://github.com/yaoyao-liu/mnemonics/issues/new) or send me an email. 
<br>
Email address: yaoyao.liu (at) mpi-inf.mpg.de

#### Summary

* [Introduction](#introduction)
* [Installation](#installation)
* [Running Experiments](#running-experiments)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Introduction

Multi-Class Incremental Learning (MCIL) aims to learn new concepts by incrementally updating a model trained on previous concepts. However, there is an inherent trade-off to effectively learning new concepts without catastrophic forgetting of previous ones. To alleviate this issue, it has been proposed to keep around a few examples of the previous concepts but the effectiveness of this approach heavily depends on the representativeness of these examples. This paper proposes a novel and automatic framework we call *mnemonics*, where we parameterize exemplars and make them optimizable in an end-to-end manner. We train the framework through bilevel optimizations, i.e., model-level and exemplar-level. We conduct extensive experiments on three MCIL benchmarks, CIFAR-100, ImageNet-Subset and ImageNet, and show that using mnemonics exemplars can surpass the state-of-the-art by a large margin. Interestingly and quite intriguingly, the mnemonics exemplars tend to be on the boundaries between classes.


<p align="center">
    <img src="https://yyliu.net/images/misc/mnemonics.png" width="600"/>
</p>

> Figure: The t-SNE results of three exemplar methods in two phases. The original data of 5 colored classes occur in the early phase. In each colored class, deep-color points are exemplars, and light-color ones show the original data as reference of the real data distribution. Gray crosses represent other participating classes, and each cross for one class. We have two main observations. (1) Our approach results in much clearer separation in the data, than random (where exemplars are randomly sampled in the early phase) and herding (where exemplars are nearest neighbors of the mean sample in the early phase). (2) Our learned exemplars mostly locate on the boundaries between classes.

## Project Architecture

```
.
├── models                             # model files
|   ├── modified_linear.py             # modified liner class
|   ├── modified_resnet_cifar.py       # modified resnet class
|   └── modified_resnetmtl_cifar.py    # modified resnet with transferring weights class
├── trainer                            # trianer files  
|   ├── incremental_train_and_eval.py  # incremental learning for one phase
|   ├── mnemonics_train.py             # mnemonics training class
|   └── train.py                       # main trainer class
├── utils                              # a series of tools used in this repo
|   ├── compute_accuracy.py            # function for computing accuracy
|   ├── compute_features.py            # function for computing features
|   ├── conv2d_mtl.py                  # transferring weights tools
|   ├── gpu_tools.py                   # GPU tools
|   ├── misc.py                        # miscellaneous tool functions
|   └── process_mnemonics.py           # function for processing mnemonics exemplars
└── main.py                            # the python file with main function and parameter settings
```

## Installation

In order to run this repository, we advise you to install python 3.6 and PyTorch 0.4.0 with Anaconda.
You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:
```bash
conda create --name mnemonics-pytorch python=3.6
conda activate mnemonics-pytorch
conda install pytorch=0.4.0 
conda install torchvision -c pytorch
```

Install other requirements:
```bash
pip install -r requirements.txt
```

## Running Experiments

Run the experiment using the following command:
```bash
python main.py --nb_cl_fg=50 --nb_cl=2 --nb_protos 20 --resume --imprint_weights
```

## Citation

Please cite our paper if it is helpful to your work:

```
@inproceedings{liu2020mnemonics,
author = {Liu, Yaoyao and Su, Yuting and Liu, An{-}An and Schiele, Bernt and Sun, Qianru},
title = {Mnemonics Training: Multi-Class Incremental Learning without Forgetting},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:

* [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)

* [iCaRL: Incremental Classifier and Representation Learning](https://github.com/srebuffi/iCaRL)
