<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>





[pypi-image]:https://badge.fury.io/py/dive-into-graphs.svg
[pypi-url]:https://pypi.org/project/dive-into-graphs/
[docs-image]: https://readthedocs.org/projects/diveintographs/badge/?version=latest
[docs-url]: https://diveintographs.readthedocs.io/en/latest/?badge=latest
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg
[license-url]:https://github.com/divelab/DIG/blob/main/LICENSE
[contributor-image]:https://img.shields.io/github/contributors/divelab/DIG
[contributor-url]:https://github.com/divelab/DIG/graphs/contributors 
[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]:https://diveintographs.readthedocs.io/en/latest/contribution/instruction.html

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]
[![Build Status](https://travis-ci.com/divelab/DIG.svg?branch=dig)](https://travis-ci.com/divelab/DIG)
[![codecov](https://codecov.io/gh/divelab/DIG/branch/dig/graph/badge.svg?token=KBJ1P31VCH)](https://codecov.io/gh/divelab/DIG)
![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]
<!--- [![Contributors][contributor-image]][contributor-url] -->


**[Documentation](https://diveintographs.readthedocs.io)** | **[Paper](https://arxiv.org/abs/2103.12608)** | **[Benchmarks/Examples](https://github.com/divelab/DIG/tree/dig/benchmarks)** | **[Tutorials](https://diveintographs.readthedocs.io/en/latest/tutorials/graphdf.html#)**

*DIG: Dive into Graphs* is a turnkey library for graph deep learning research.


## Why DIG?

The key difference with current graph deep learning libraries, such as PyTorch Geometric (PyG) and Deep Graph Library (DGL), is that, while PyG and DGL support basic graph deep learning operations, DIG provides a unified testbed for higher level, research-oriented graph deep learning tasks, such as graph generation, self-supervised learning, explainability, and 3D graphs.

If you are working or plan to work on research in graph deep learning, DIG enables you to develop your own methods within our extensible framework, and compare with current baseline methods using common datasets and evaluation metrics without extra efforts.

## Overview

It includes unified implementations of **data interfaces**, **common algorithms**, and **evaluation metrics** for several advanced tasks. Our goal is to enable researchers to easily implement and benchmark algorithms. Currently, we consider the following research directions.

* **Graph Generation**: `dig.ggraph`
* **Self-supervised Learning on Graphs**: `dig.sslgraph`
* **Explainability of Graph Neural Networks**: `dig.xgraph`
* **Deep Learning on 3D Graphs**: `dig.threedgraph`



<p align="center">
<img src="https://github.com/divelab/DIG/blob/dig/imgs/DIG-overview.png" width="700" class="center" alt="logo"/>
    <br/>
</p>



## Installation

### Install from pip
The key dependencies of DIG: Dive into Graphs are PyTorch (>=1.6.0), PyTorch Geometric (>=1.6.0), and RDKit.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) (>=1.6.0)

```shell script
$ python -c "import torch; print(torch.__version__)"
>>> 1.6.0
```

2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) (>=1.6.0)

```shell script
$ python -c "import torch_geometric; print(torch_geometric.__version__)"
>>> 1.6.0
```

    
3. Install DIG: Dive into Graphs.

```shell script
pip install dive-into-graphs
```


After installation, you can check the version. You have successfully installed DIG: Dive into Graphs if no error occurs.

``` shell script
$ python
>>> from dig.version import __version__
>>> print(__version__)
```

### Install from source
If you want to try the latest features that have not been released yet, you can install dig from source.

```shell script
git clone https://github.com/divelab/DIG.git
cd DIG
pip install .
```


## Usage

For details of all included APIs, please refer to the [documentation](https://diveintographs.readthedocs.io/). We also provide [benchmark implementations](https://github.com/divelab/DIG/tree/dig/benchmarks) as examples to use APIs provided in *DIG*. You can get started with your interested directions by clicking the following links.

* [Graph Generation](https://github.com/divelab/DIG/tree/dig/benchmarks/ggraph): [`JT-VAE`](https://github.com/divelab/DIG/tree/dig/benchmarks/ggraph/JTVAE), [`GraphAF`](https://github.com/divelab/DIG/tree/dig/benchmarks/ggraph/GraphAF), [`GraphDF`](https://github.com/divelab/DIG/tree/dig/benchmarks/ggraph/GraphDF), [`GraphEBM`](https://github.com/divelab/DIG/tree/dig/benchmarks/ggraph/GraphEBM).
* [Self-supervised Learning on Graphs](https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph): [`InfoGraph`](https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_infograph.ipynb), [`GRACE`](https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_grace.ipynb), [`MVGRL`](https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_mvgrl.ipynb), [`GraphCL`](https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_graphcl.ipynb).
* [Explainability of Graph Neural Networks](https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph): [`DeepLIFT`](https://github.com/divelab/DIG/blob/dig/benchmarks/xgraph/deeplift.ipynb), [`GNN-LRP`](https://github.com/divelab/DIG/blob/dig/benchmarks/xgraph/gnn_lrp.ipynb), [`GNNExplainer`](https://github.com/divelab/DIG/blob/dig/benchmarks/xgraph/gnnexplainer.ipynb), [`GradCAM`](https://github.com/divelab/DIG/blob/dig/benchmarks/xgraph/gradcam.ipynb), [`PGExplainer`](https://github.com/divelab/DIG/blob/dig/benchmarks/xgraph/pgexplainer.ipynb), [`SubgraphX`](https://github.com/divelab/DIG/blob/dig/benchmarks/xgraph/subgraphx.ipynb).
* [Deep Learning on 3D Graphs](https://github.com/divelab/DIG/tree/dig/benchmarks/threedgraph): [`SchNet`](https://github.com/divelab/DIG/blob/dig/benchmarks/threedgraph/threedgraph.ipynb), [`DimeNet++`](https://github.com/divelab/DIG/blob/dig/benchmarks/threedgraph/threedgraph.ipynb), [`SphereNet`](https://github.com/divelab/DIG/blob/dig/benchmarks/threedgraph/threedgraph.ipynb).


## Contributing

We welcome any forms of contributions, such as reporting bugs and adding new features. Please refer to our [contributing guidelines](https://diveintographs.readthedocs.io/en/latest/contribution/instruction.html) for details.


## Citing DIG

Please cite our [paper](https://arxiv.org/abs/2103.12608) if you find *DIG* useful in your work:
```
@article{liu2021dig,
      title={{DIG}: A Turnkey Library for Diving into Graph Deep Learning Research}, 
      author={Meng Liu and Youzhi Luo and Limei Wang and Yaochen Xie and Hao Yuan and Shurui Gui and Haiyang Yu and Zhao Xu and Jingtun Zhang and Yi Liu and Keqiang Yan and Haoran Liu and Cong Fu and Bora Oztekin and Xuan Zhang and Shuiwang Ji},
      journal={arXiv preprint arXiv:2103.12608},
      year={2021},
}
```

## The Team

*DIG: Dive into Graphs* is developed by [DIVE](https://github.com/divelab/)@TAMU. Contributors are Meng Liu*, Youzhi Luo*, Limei Wang*, Yaochen Xie*, Hao Yuan*, Shurui Gui*, Haiyang Yu*, Zhao Xu, Jingtun Zhang, Yi Liu, Keqiang Yan, Haoran Liu, Cong Fu, Bora Oztekin, Xuan Zhang, and Shuiwang Ji.

## Contact

If you have any technical questions, please submit new issues.

If you have any other questions, please contact us: Meng Liu [mengliu@tamu.edu] and Shuiwang Ji [sji@tamu.edu].


