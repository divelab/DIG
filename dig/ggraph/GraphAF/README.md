# GraphAF

This is a re-implementation for [GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation(https://arxiv.org/abs/2001.09382).]

![](https://github.com/divelab/DIG/blob/main/dig/ggraph/GraphAF/figs/graphaf.png)


## Table of Contents

1. [Setup](#setup)
1. [Usage](#usage)
1. [Citation](#citation)
1. [Acknowledgement](#acknowledgement)



## Setup

To install the conda virtual environment `graphaf`:
```shell script
$ cd /ggraph/GraphAF
$ bash install.sh
```
Note that we use CUDA 10.1 in this project. If you have other CUDA versions, you should install the PyTorch and cudatoolkit compatible with your CUDA.


## Usage

### Random Generation

You can use our trained models in `GraphAF/ckpt/dense_gen_net_10.pth` or train the model from scratch, you can change all the experimental settings in `GraphAF/config/dense_gen_config.py`:
```shell script
$ cd GraphAF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python main_density.py 
```
To generate molecules using our trained model, you can open the jupyter notebook and use the example in `GraphAF/density_test.ipynb`:
```shell script
$ cd GraphAF
$ jupyter notebook
```

### Property Optimization

For property optimization, we aim to generate molecules with desirable properties (*i.e.*, QED and plogp in this work). You can use our trained models in `GraphAF/prop_optim` or train the model using the official pretrained model `GraphAF/ckpt/checkpoint277` by reinforcement learning, you can change all the experimental settings in `GraphAF/config/prop_optim_config.py`:
```shell script
$ cd GraphAF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python main_prop_optim.py 
```

To generate molecules using our trained model, you can open the jupyter notebook and use the example in `GraphAF/prop_test.ipynb`:
```shell script
$ cd GraphAF
$ jupyter notebook
```

### Constrained Optimization

For constrained optimization, we aim to optimize molecules with desirable properties (plogp in this work). You can use our trained models in `GraphAF/cons_optim` or train the model using the official pretrained model `GraphAF/ckpt/checkpoint277` by reinforcement learning, you can change all the experimental settings in `GraphAF/config/con_optim_config.py`:
```shell script
$ cd GraphAF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python main_constrained_optim.py
```

To optimize molecules using our trained model, you can open the jupyter notebook and use the example in `GraphAF/cons_optim_test.ipynb`:
```shell script
$ cd GraphAF
$ jupyter notebook
```
### Citation
```
@article{shi2020graphaf,
  title={{GraphAF}: a Flow-based Autoregressive Model for Molecular Graph Generation},
  author={Chence Shi and Minkai Xu and Zhaocheng Zhu and Weinan Zhang and Ming Zhang and Jian Tang},
  journal={iclr},
  year={2020}
}
```

### Acknowledgement
Our implementation is based on [GraphAF](https://github.com/DeepGraphLearning/GraphAF). Thanks a lot for their awesome works.
