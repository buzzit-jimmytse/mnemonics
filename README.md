## Mnemonics Training: Multi-Class Incremental Learning without Forgetting

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

### Code will be released soon.

This repository contains the PyTorch implementation for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "[Mnemonics Training: Multi-Class Incremental Learning without Forgetting](https://arxiv.org/pdf/2002.10211.pdf)" by [Yaoyao Liu](https://yyliu.net/), An-An Liu, Yuting Su, [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/), and [Qianru Sun](https://qianrusun1015.github.io).

If you have any questions on this repository or the related paper, feel free to create an issue or send me an email. 
<br>
Email address: yaoyao.liu (at) mpi-inf.mpg.de

## Introduction

Multi-Class Incremental Learning (MCIL) aims to learn new concepts by incrementally updating a model trained on previous concepts. However, there is an inherent trade-off to effectively learning new concepts without catastrophic forgetting of previous ones. To alleviate this issue, it has been proposed to keep around a few examples of the previous concepts but the effectiveness of this approach heavily depends on the representativeness of these examples. This paper proposes a novel and automatic framework we call *mnemonics*, where we parameterize exemplars and make them optimizable in an end-to-end manner. We train the framework through bilevel optimizations, i.e., model-level and exemplar-level. We conduct extensive experiments on three MCIL benchmarks, CIFAR-100, ImageNet-Subset and ImageNet, and show that using mnemonics exemplars can surpass the state-of-the-art by a large margin. Interestingly and quite intriguingly, the mnemonics exemplars tend to be on the boundaries between classes.


<p align="center">
    <img src="https://yyliu.net/images/misc/mnemonics.png" width="400"/>
</p>

> Figure: The t-SNE results of three exemplar methods in two phases. The original data of 5 colored classes occur in the early phase. In each colored class, deep-color points are exemplars, and light-color ones show the original data as reference of the real data distribution. Gray crosses represent other participating classes, and each cross for one class. We have two main observations. (1) Our approach results in much clearer separation in the data, than random (where exemplars are randomly sampled in the early phase) and herding (where exemplars are nearest neighbors of the mean sample in the early phase). (2) Our learned exemplars mostly locate on the boundaries between classes.

## Citation

Please cite our paper if it is helpful to your work:

```
@inproceedings{liu2020mnemonics,
author = {Liu, Yaoyao and Su, Yuting and Liu, An{-}An and Schiele, Bernt and Sun, Qianru},
title = {Mnemonics Training: Multi-Class Incremental Learning without Forgetting},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
