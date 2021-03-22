# GNNExplainer

Code for GNNExplainer following the [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894).

## Table of Contents

1. [Requirements](#requirements)
1. [Installation](#installation)
1. [Usage](#usage)
1. [Demos](#demos)
1. [Citation](#citation)

## Requirements

* Ubuntu
* Anaconda
* Cuda 10.2 & Cudnn (>=7.0)

## Installation

* Clone this repo
* Install the conda environment `xgraph`
* Download datasets, then download the pretrained models.

```shell script
$ git clone git@github.com:divelab/DIG.git (or directly clone the xgraph directory by svn)
$ cd DIG/dig/xgraph/GNNExplainer
$ source ./install.bash
```
Download [Datasets](https://mailustceducn-my.sharepoint.com/:u:/g/personal/agnesgsr_mail_ustc_edu_cn/EdH7QVBBghBBgmMgf0_UZSAByxkMa3AvRdH7_QwD9MUfrw?e=EN3JiS) to `xgraph/datasets/`, then
download [pre-trained models](https://mailustceducn-my.sharepoint.com/:u:/g/personal/agnesgsr_mail_ustc_edu_cn/EZklsgM56i5EtCKeeEpTTLIBNpDvDNB-zol6ROXBngPsZg?e=20IBOg) to `xgraph/GNNExplainer/`
```shell script
$ cd GNNExplainer 
$ unzip ../datasets/datasets.zip -d ../datasets/
$ unzip checkpoints.zip
```

## Usage

For running GNNExplainer on the given model and the dataset with the first 100 data:

```shell script
python -m benchmark.kernel.pipeline --task explain --model_name [GCN_2l/GCN_3l/GIN_2l/GIN_3l] --dataset_name [ba_shape/ba_lrp/tox21/clintox] --target_idx [0/2] --explainer GNNExplainer --sparsity [0.5/...]
```

For running GNNExplainer with the given data, please add the flag `--debug`, then modify the index at line xx in `benchmark/kernel/pipeline.py` to choose your data in the dataset. Please add the flag `--vis` for important edges visualization while add one more flag `--walk` to visualize the flow view.

Note that the 2-layer models GCN_2l and GIN_3l only work on dataset ba_shape, while 3-layer models work on the left three datasets. Specially, the tox21's target_idx is 2 while others are 0. You can choose any sparsity between 0 to 1 as you like. Higher sparsity means less important edges to be chosen.

If you want to save the visualization result in debug mode, please use `--save_fig` flag. Then the output figure will be saved
in the `./visual_results/` folder.

## Demos

We provide a visualization example:

```bash
python -m benchmark.kernel.pipeline --task explain --model_name GCN_3l --dataset_name clintox --target_idx 0 --explainer GNNExplainer --sparsity 0.5 --debug --vis --nolabel
```
where the `--nolabel` means to remove debug labels on the graph.

The edge view of the example on clintox is:

<img src="./figures/clintox.png" alt="ba_shape_edge" style="zoom:30%"/>

where F means Fidelity while S means Sparsity.

## Citation

If using our implementation, please cite our work.

```
@article{yuan2020explainability,
  title={Explainability in Graph Neural Networks: A Taxonomic Survey},
  author={Yuan, Hao and Yu, Haiyang and Gui, Shurui and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.15445},
  year={2020}
}
```

