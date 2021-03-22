# SchNet

This is an re-implementation for [SchNet: A continuous-filter convolutional neural net work for modeling quantum interactions](https://arxiv.org/abs/1706.08566) under the 3DGN framework.

## Table of Contents

1. [Setup](#setup)
1. [Usage](#usage)
1. [Citation](#citation)
1. [Acknowledgement](#acknowledgement)


## Setup

To install the conda virtual environment `schnet`:
```shell script
$ cd /3dgraph/schnet
$ bash setup.sh
```
Note that we use CUDA 10.1 in this project. If you have other CUDA versions, you should install the PyTorch and cudatoolkit compatible with your CUDA. Note that the versions of PyTorch and PyTorch Geometric should be compatible. It would be easy to install PyTorch Geometric by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).


## Usage
```shell script
cd 3dgraph/schnet
CUDA_VISIBLE_DEVICES=${your_gpu_id} python main.py --save_dir $save_dir --dataset $dataset --target $target --batch_size $batch_size --epochs $epochs 
```
## Citations

```
@article{schutt2017schnet,
  title={Schnet: A continuous-filter convolutional neural network for modeling quantum interactions},
  author={Sch{\"u}tt, Kristof T and Kindermans, Pieter-Jan and Sauceda, Huziel E and Chmiela, Stefan and Tkatchenko, Alexandre and M{\"u}ller, Klaus-Robert},
  journal={arXiv preprint arXiv:1706.08566},
  year={2017}
}
```

## Acknowledgement
Our implementation is based on [Models in PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py). Thanks a lot for their awesome works.