# Graph Generation

## Overview

The ggraph package is a collection of benchmark datasets, data interfaces, evaluation metrics, and state-of-the-art algorithms for graph generation. We aims to provide standardized datasets and unified performance evaluation for academic researchers interested in graph generation. We cover the following three tasks:

1. **Random generation task**, learning a generative model to capture the probability density of a dataset;
1. **Property optimization task** (or goal-directed generation task), learning a biased generative model to generate molecular graphs with desired properties;
1. **Constrained optimization task**, learning a graph translation model to optimize the desired property of the input molecular graphs.

## Implemented Algorithms

The `ggraph` package implements four state-of-the-art graph generation algorithms and offers detailed code running instructions. The information about the four algorithms is summarized in the following table.

| Method | Links | Brief description |
| ------ | ----- | ------------------ |
| JT-VAE | [Paper](https://arxiv.org/abs/1802.04364) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/ggraph/JT-VAE) | JT-VAE is a graph generation model based on variational autoencoder. It generates molecular graphs by first forming a tree-structured scaffold over chemical substructures, then combining them into a molecular graph. |
| GraphAF | [Paper](https://arxiv.org/abs/2001.09382) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphAF) | GraphAF is a sequential generation method based on normalizing flow models. It adopts masked autoregressive flow model to iteratively sampling new graph nodes and edges.|
| GraphEBM | [Paper](https://arxiv.org/abs/2102.00546) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphEBM) | GraphEBM is a one-shot generation method based on Energy-Based models. It perserves the intrinsic property of permutation invariant by parameterize the energy function in a permutation invariant manner.|
| GraphDF | [Paper](https://arxiv.org/abs/2102.01189) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphDF) | GraphDF is a sequential generation method based on discrete flow models. It uses invertible modulo shift transforms to iterately map discrete latent variables to graph nodes and edges. |

## Package Usage

Here we provide some examples of using the unified data interfaces and evaluation metrics.

(1) Data interfaces

We provide unified data interfaces for reading benchmark datasets and a standard Pytorch data loader. Molecules are processed into graphs, and loaded in the form of atom type matrices and adjacency tensors.

```python
from utils import get_smiles_zinc250k, MolSet
from torch.utils.data import DataLoader

smiles = get_smiles_qm9('./datasets/zinc250k.csv')
dataset = MolSet(smile_list=smiles)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

(2) Evaluation metrics

We also provide unified evaluation metrics for easy evaluation of three graph generation tasks. The evaluation metric functions take the list of rdkit molecule objects as the input, and returns a dictionary storing the metric values. See the comments in utils/metric.py for the detailed usage instructions.

```python
from utils import metric_random_generation, metric_property_optimization, metric_constrained_optimization

rg_metrics = metric_random_generation(rg_mols, dataset_smiles)
po_metrics = metric_property_optimization(plogp_mols, topk=3, prop='plogp')
co_metrics = metric_constrained_optimization(co_mols_0, co_mols_2, co_mols_4, co_mols_6, '../datasets/zinc_800_jt.csv')
```

## Contact
*If you have any questions, please submit a new issue or contact us at Youzhi Luo [yzluo@tamu.edu] and Shuiwang Ji [sji@tamu.edu] .*
