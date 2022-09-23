# Task-Agnostic GNN Explainer: Universal Explanation via Contrastive Learning

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