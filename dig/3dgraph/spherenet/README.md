# SphereNet

This is an official implementation for [Spherical Message Passing for 3D Graph Networks](https://arxiv.org/abs/2102.05013v2) under the 3DGN framework.

![](https://github.com/divelab/DIG/blob/main/dig/3dgraph/spherenet/figs/sphere.png)


## Table of Contents

1. [Setup](#setup)
1. [Usage](#usage)
1. [Citation](#citation)
1. [Acknowledgement](#acknowledgement)


## Setup

To install the conda virtual environment `spherenet`:
```shell script
$ cd /3dgraph/spherenet
$ bash setup.sh
```
Note that we use CUDA 10.1 in this project. If you have other CUDA versions, you should install the PyTorch and cudatoolkit compatible with your CUDA. Note that the versions of PyTorch and PyTorch Geometric should be compatible. It would be easy to install PyTorch Geometric by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).


## Usage
```shell script
cd 3dgraph/spherenet
CUDA_VISIBLE_DEVICES=${your_gpu_id} python main.py --save_dir $save_dir --dataset $dataset --target $target --batch_size $batch_size --epochs $epochs 
```
## Citation

```
@article{liu2021spherical,
  title={Spherical Message Passing for 3D Graph Networks},
  author={Liu, Yi and Wang, Limei and Liu, Meng and Zhang, Xuan and Oztekin, Bora and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2102.05013},
  year={2021}
}

```
## Acknowledgements
Our implementation is based on [DimeNet](https://github.com/klicperajo/dimenet) and [Models in PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py). Thanks a lot for their awesome works.
