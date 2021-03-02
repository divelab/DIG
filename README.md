<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>

------

# DIG: Dive into Graphs
*DIG: Dive into Graphs* is a research-oriented library for geometric (*a.k.a.*, graph) deep learning. Please refer to this [doc](https://docs.google.com/document/d/1FfpXGiP1dkRf6BFpXmF2cXzyAGd9o5SFuaJk7rnjPnk/edit?usp=sharing) for development instructions.

It includes various methods in several significant research directions for geometric deep learning. Our goal is to enable researchers to rerun baselines and implement their new ideas, conveniently. Currently, it consists of the following topics:

* [Generation](#generation)
* [Self-supervised Learning](#self-supervised-learning)
* [Interpretability](#interpretability)
* [3D](#3d)

## Generation
In [`ggraph`](https://github.com/divelab/DIG/tree/main/dig/ggraph), the following methods are included:

* [`GCPN`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GCPN) from [Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation](https://arxiv.org/abs/1806.02473)
* [`JTVAE`](https://github.com/divelab/DIG/tree/main/dig/ggraph/JT-VAE) from [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/pdf/1802.04364])
* [`GraphAF`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphAF) from [GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation](https://arxiv.org/abs/2001.09382)
* [`GraphDF`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphDF) from [GraphDF: A Discrete Flow Model for Molecular Graph Generation](https://arxiv.org/abs/2102.01189)
* [`GraphEBM`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphEBM) from [GraphEBM: Molecular Graph Generation with Energy-Based Models](https://arxiv.org/abs/2102.00546)


## Self-supervised Learning



## Interpretability
In [`xgraph`](https://github.com/divelab/DIG/tree/main/dig/xgraph), the following methods are included:
* [`XGNN`](https://github.com/divelab/DIG/tree/main/dig/xgraph/XGNN) from [XGNN: Towards Model-Level Explanations of Graph Neural Networks](https://arxiv.org/abs/2006.02587)
* [`GNN-LRP`](https://github.com/divelab/DIG/tree/main/dig/xgraph/GNN-LRP) from [Higher-Order Explanations of Graph Neural Networks via Relevant Walks](https://arxiv.org/abs/2006.03589)
* [`SubgraphX`](https://github.com/divelab/DIG/tree/main/dig/xgraph/SubgraphX) from [On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/abs/2102.05152)

## 3D
In [`3dgraph`](https://github.com/divelab/DIG/tree/main/dig/3dgraph), the following methods are included:

* [`DimeNet++`](https://github.com/divelab/DIG/tree/main/dig/3dgraph/dimenetpp) from [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115)
* [`SphereNet`](https://github.com/divelab/DIG/tree/main/dig/3dgraph/spherenet) from [spherical message passing for 3d graph networks](https://arxiv.org/abs/2102.05013v2)



------
