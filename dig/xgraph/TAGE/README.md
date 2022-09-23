# Task-Agnostic Graph Explanations

This is the official implementation of the paper *Task-Agnostic Graph Explanations* in NeurIPS 2022.

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

@inproceedings{xie2022task,
  title={Task-Agnostic Graph Explanations},
  author={Xie, Yaochen and Katariya, Sumeet and Tang, Xianfeng and Huang, Edward and Rao, Nikhil and Subbian, Karthik and Ji, Shuiwang},
  booktitle={The 36th Annual Conference on Neural Information Processing Systems},
  year={2022}
}