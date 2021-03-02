# SphereNet and its special cases, including DimeNet++, DimeNet, etc.

This includes the official implementation for [Spherical Message Passing for 3D Graph Networks](https://arxiv.org/abs/2102.05013v2), and re-implementation for [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115) under the 3DGN framewrok.

![](https://github.com/divelab/DIG/blob/main/dig/3dgraph/smp/figs/sphere.png)


## Table of Contents

1. [Requirements](#requirements)
1. [Installation](#installation)
1. [Usage](#usage)
1. [Citation](#citation)


## Requirements

* PyTorch
* PyTorch Geometric >= 1.3.1
* NetworkX
* tdqm


Note that the versions of PyTorch and PyTorch Geometric should be compatible and PyTorch Geometric is related to other packages, which need to be installed in advance. It would be easy by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).


## Installation

* Clone this repo
* Install the conda environment `3dgraph`
* Download datasets, then download the pretrained models.

```shell script
$ git clone git@github.com:divelab/DIG.git
$ cd DIG/dig/3dgraph/spherenet
$ source ./install.bash
```
Download [Datasets](https://mailustceducn-my.sharepoint.com/:u:/g/personal/agnesgsr_mail_ustc_edu_cn/Ebwg9j6YHPJDh5nZKrd4x6UBMvz2kJMw2y3wgp8GNLYOVw?e=3cILKu) to `3dgraph/datasets/`, then
download [pre-trained models](https://mailustceducn-my.sharepoint.com/:u:/g/personal/agnesgsr_mail_ustc_edu_cn/ERQCHDEHnq5DiW-XHyiP5C0BE2taSyEmzX_PLwQolMTkkA?e=y6mqtV) to `3dgraph/spherenet/`
```shell script
$ cd spherenet 
$ unzip ../datasets/datasets.zip -d ../datasets/
$ unzip checkpoints.zip
```

## Usage


## Citation

```
@article{liu2021spherical,
  title={Spherical Message Passing for 3D Graph Networks},
  author={Liu, Yi and Wang, Limei and Liu, Meng and Zhang, Xuan and Oztekin, Bora and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2102.05013},
  year={2021}
}
```
