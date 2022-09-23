# Task-Agnostic Graph Explanations

This is the official implementation of the paper [*"Task-Agnostic Graph Explanations"*](https://arxiv.org/abs/2202.08335) appears in NeurIPS 2022.

<p align="center">
    <br/>
    <img src="https://github.com/divelab/DIG/blob/main/dig/xgraph/TAGE/pipeline.jpg" width="900" class="center" alt="task-agnostic"/>
    <br/>
</p>

## Abstract

Graph Neural Networks (GNNs) have emerged as powerful tools to encode graph-structured data. Due to their broad applications, there is an increasing need to develop tools to explain how GNNs make decisions given graph-structured data. Existing learning-based GNN explanation approaches are task-specific in training and hence suffer from crucial drawbacks. Specifically, they are incapable of producing explanations for a multitask prediction model with a single explainer. They are also unable to provide explanations in cases where the GNN is trained in a self-supervised manner, and the resulting representations are used in future downstream tasks. To address these limitations, we propose a Task-Agnostic GNN Explainer (TAGE) that is independent of downstream models and trained under self-supervision with no knowledge of downstream tasks. TAGE enables the explanation of GNN embedding models with unseen downstream tasks and allows efficient explanation of multitask models. Our extensive experiments show that TAGE can significantly speed up the explanation efficiency by using the same model to explain predictions for multiple downstream tasks while achieving explanation quality as good as or even better than current state-of-the-art GNN explanation approaches. 

## Environment Requirements
- jupyter
- pytorch
- pytorch-geometric
- rdkit
- dig

You can follow the instructions of [dig](https://github.com/divelab/DIG) to install compatible version of pytorch and pytorch-geometric.


## Usage

The public datasets used in this work are MoleculeNet and PPI. MoleculeNet can be downloaded from [here](https://github.com/snap-stanford/pretrain-gnns#dataset-download). You can move the `dataset` folder to the root directory of the repo.

We have provided trained GNN models to be explained. The trained explainers are also provided. You can follow the examples in `gexplain_2stage_quant.ipynb` and `nexplain_2stage_quant.ipynb` to reproduce the results. The visualizations appear in the paper can be reproduced by running `gexplain_2stage_visual_[bace/hiv/sider].ipynb`. Results on the synthetic dataset, BAShapes, can be reproduced by running `syn_dataset.ipynb`.

## Bibtex

If you use this code, please cite the paper.
```
@inproceedings{xie2022task,
  title={Task-Agnostic Graph Explanations},
  author={Xie, Yaochen and Katariya, Sumeet and Tang, Xianfeng and Huang, Edward and Rao, Nikhil and Subbian, Karthik and Ji, Shuiwang},
  booktitle={The 36th Annual Conference on Neural Information Processing Systems},
  year={2022}
}
```
