# DimeNet++

This is an re-implementation for [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115) under the 3DGN framework.

## Table of Contents

1. [Setup](#setup)
1. [Usage](#usage)
1. [Citation](#citation)
1. [Acknowledgement](#acknowledgement)


## Setup

To install the conda virtual environment `dimenetpp`:
```shell script
$ cd /3dgraph/dimenetpp
$ bash setup.sh
```
Note that we use CUDA 10.1 in this project. If you have other CUDA versions, you should install the PyTorch and cudatoolkit compatible with your CUDA. Note that the versions of PyTorch and PyTorch Geometric should be compatible. It would be easy to install PyTorch Geometric by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).


## Usage
```shell script
cd 3dgraph/dimenetpp
CUDA_VISIBLE_DEVICES=${your_gpu_id} python main.py --save_dir $save_dir --dataset $dataset --target $target --batch_size $batch_size --epochs $epochs 
```
## Citations

```
@inproceedings{klicpera_dimenet_2020,
  title = {Directional Message Passing for Molecular Graphs},
  author = {Klicpera, Johannes and Gro{\ss}, Janek and G{\"u}nnemann, Stephan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year = {2020}
}
```
```
@inproceedings{klicpera_dimenetpp_2020,
title = {Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules},
author = {Klicpera, Johannes and Giri, Shankari and Margraf, Johannes T. and G{\"u}nnemann, Stephan},
booktitle={NeurIPS-W},
year = {2020}
}
```

## Acknowledgement
Our implementation is based on [DimeNet](https://github.com/klicperajo/dimenet) and [Models in PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py). Thanks a lot for their awesome works.