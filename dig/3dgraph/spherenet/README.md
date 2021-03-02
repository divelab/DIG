# SphereNet

This is the official implementation for [Spherical Message Passing for 3D Graph Networks](https://arxiv.org/abs/2102.05013v2).

![](https://github.com/divelab/DIG/blob/main/dig/3dgraph/spherenet/figs/sphere.png)


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
* Install the conda environment `xgraph`
* Download datasets, then download the pretrained models.

```shell script
$ git clone git@github.com:divelab/DIG.git
$ cd DIG/dig/xgraph/GNN-LRP
$ source ./install.bash
```
Download [Datasets](https://mailustceducn-my.sharepoint.com/:u:/g/personal/agnesgsr_mail_ustc_edu_cn/Ebwg9j6YHPJDh5nZKrd4x6UBMvz2kJMw2y3wgp8GNLYOVw?e=3cILKu) to `xgraph/datasets/`, then
download [pre-trained models](https://mailustceducn-my.sharepoint.com/:u:/g/personal/agnesgsr_mail_ustc_edu_cn/ERQCHDEHnq5DiW-XHyiP5C0BE2taSyEmzX_PLwQolMTkkA?e=y6mqtV) to `xgraph/GNN-LRP/`
```shell script
$ cd GNN-LRP 
$ unzip ../datasets/datasets.zip -d ../datasets/
$ unzip checkpoints.zip
```

## Usage

For running GNN-LRP or GNN-GI on the given model and the dataset with the first 100 data:

```shell script
python -m benchmark.kernel.pipeline --task explain --model_name [GCN_2l/GCN_3l/GIN_2l/GIN_3l] --dataset_name [ba_shape/ba_lrp/tox21/clintox] --target_idx [0/2] --explainer [GNN_LRP/GNN_GI] --sparsity [0.5/...]
```

For running GNN-LRP or GNN-GI with the given data, please add the flag `--debug`, then modify the index at line xx in `benchmark/kernel/pipeline.py` to choose your data in the dataset. Please add the flag `--vis` for important edges visualization while add one more flag `--walk` to visualize the flow view.

Note that the 2-layer models GCN_2l and GIN_3l only work on dataset ba_shape, while 3-layer models work on the left three datasets. Specially, the tox21's target_idx is 2 while others are 0. You can choose any sparsity between 0 to 1 as you like. Higher sparsity means less important edges to be chosen.

If you want to save the visualization result in debug mode, please use `--save_fig` flag. Then the output figure will be saved
in the `./visual_results/` folder.


## Citation

```
@article{liu2021spherical,
  title={Spherical Message Passing for 3D Graph Networks},
  author={Liu, Yi and Wang, Limei and Liu, Meng and Zhang, Xuan and Oztekin, Bora and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2102.05013},
  year={2021}
}
```
