<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>

------

# DIG: Dive into Graphs
*DIG: Dive into Graphs* is a research-oriented library for geometric (*a.k.a.*, graph) deep learning. Please refer to this [doc](https://docs.google.com/document/d/1FfpXGiP1dkRf6BFpXmF2cXzyAGd9o5SFuaJk7rnjPnk/edit?usp=sharing) for development instructions.

It includes various methods in several significant research directions for geometric deep learning. Our goal is to enable researchers to rerun baselines and implement their new ideas, conveniently. Currently, it consists of the following topics:

* [Graph Generation](#graph-generation)
* [Self-supervised Learning on Graphs](#self-supervised-learning-on-graphs)
* [Interpretability of Graph Neural Networks](#interpretability-of-graph-neural-networks)
* [Deep Learning on 3D Graphs](#deep-learning-on-3d-graphs)

## Graph Generation
In [`ggraph`](https://github.com/divelab/DIG/tree/main/dig/ggraph), the following methods are included:

* [`GCPN`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GCPN) from [Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation](https://arxiv.org/abs/1806.02473)
* [`JTVAE`](https://github.com/divelab/DIG/tree/main/dig/ggraph/JT-VAE) from [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/pdf/1802.04364])
* [`GraphAF`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphAF) from [GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation](https://arxiv.org/abs/2001.09382)
* [`GraphDF`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphDF) from [GraphDF: A Discrete Flow Model for Molecular Graph Generation](https://arxiv.org/abs/2102.01189)
* [`GraphEBM`](https://github.com/divelab/DIG/tree/main/dig/ggraph/GraphEBM) from [GraphEBM: Molecular Graph Generation with Energy-Based Models](https://arxiv.org/abs/2102.00546)


## Self-supervised Learning on Graph
In [`ssl`](https://github.com/divelab/DIG/tree/main/dig/), the following methods are included:
* [`InfoGraph`](https://github.com/divelab/DIG/tree/main/dig/) from [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](https://arxiv.org/abs/1908.01000)
* [`GRACE`](https://github.com/divelab/DIG/tree/main/dig/) from [Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131)
* [`MVGRL`](https://github.com/divelab/DIG/tree/main/dig/) from [Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/abs/2006.05582)
* [`GraphCL`](https://github.com/divelab/DIG/tree/main/dig/) from [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902)


## Interpretability of Graph Neural Networks
In [`xgraph`](https://github.com/divelab/DIG/tree/main/dig/xgraph), the following methods are included:
* [`XGNN`](https://github.com/divelab/DIG/tree/main/dig/xgraph/XGNN) from [XGNN: Towards Model-Level Explanations of Graph Neural Networks](https://arxiv.org/abs/2006.02587)
* [`GNN-LRP`](https://github.com/divelab/DIG/tree/main/dig/xgraph/GNN-LRP) from [Higher-Order Explanations of Graph Neural Networks via Relevant Walks](https://arxiv.org/abs/2006.03589)
* [`SubgraphX`](https://github.com/divelab/DIG/tree/main/dig/xgraph/SubgraphX) from [On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/abs/2102.05152)

## Deep Learning on 3D Graphs
In [`3dgraph`](https://github.com/divelab/DIG/tree/main/dig/3dgraph), the following methods are included:

* [`DimeNet++`](https://github.com/divelab/DIG/tree/main/dig/3dgraph/smp) from [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115)
* [`SphereNet`](https://github.com/divelab/DIG/tree/main/dig/3dgraph/smp) from [Spherical Message Passing for 3D Graph Networks](https://arxiv.org/abs/2102.05013v2)



------
