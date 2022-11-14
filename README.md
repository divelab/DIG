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
[![Build Status](https://travis-ci.com/divelab/DIG.svg?branch=dig-stable)](https://travis-ci.com/divelab/DIG)
[![codecov](https://codecov.io/gh/divelab/DIG/branch/dig-stable/graph/badge.svg?token=KBJ1P31VCH)](https://codecov.io/gh/divelab/DIG)
![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]
![visitors](https://visitor-badge.glitch.me/badge?page_id=jwenjian.visitor-badge)
[![Downloads](https://pepy.tech/badge/dive-into-graphs)](https://pepy.tech/project/dive-into-graphs)
<!--- [![Contributors][contributor-image]][contributor-url] -->


**[Documentation](https://diveintographs.readthedocs.io)** | **[Paper [JMLR]](https://www.jmlr.org/papers/v22/21-0343.html)** | **[Tutorials](https://diveintographs.readthedocs.io/en/latest/tutorials/graphdf.html#)** | **[Benchmarks](https://github.com/divelab/DIG/tree/dig-stable/benchmarks)** |  **[Examples](https://github.com/divelab/DIG/tree/dig-stable/examples)** | **[Join the DIG slack community now!:fire:](https://join.slack.com/t/dive-into-graphs/shared_invite/zt-1i9kn731c-RhLA1zcEGHXbIToxdVqo0g)**

*DIG: Dive into Graphs* is a turnkey library for graph deep learning research.

:fire:**Update (2022/07): We have upgraded our DIG library based on PyG 2.0.0. We recommend installing our latest version.**

## Why DIG?

The key difference with current graph deep learning libraries, such as PyTorch Geometric (PyG) and Deep Graph Library (DGL), is that, while PyG and DGL support basic graph deep learning operations, DIG provides a unified testbed for higher level, research-oriented graph deep learning tasks, such as graph generation, self-supervised learning, explainability, 3D graphs, and graph out-of-distribution.

If you are working or plan to work on research in graph deep learning, DIG enables you to develop your own methods within our extensible framework, and compare with current baseline methods using common datasets and evaluation metrics without extra efforts.

## Overview

It includes unified implementations of **data interfaces**, **common algorithms**, and **evaluation metrics** for several advanced tasks. Our goal is to enable researchers to easily implement and benchmark algorithms. Currently, we consider the following research directions.

* **Graph Generation**: `dig.ggraph`
* **Self-supervised Learning on Graphs**: `dig.sslgraph`
* **Explainability of Graph Neural Networks**: `dig.xgraph`
* **Deep Learning on 3D Graphs**: `dig.threedgraph`
* **Graph OOD**: `dig.oodgraph`


<p align="center">
<img src="https://github.com/divelab/DIG/blob/dig-stable/imgs/DIG-overview.png" width="700" class="center" alt="logo"/>
    <br/>
</p>


## Usage

Example: a few lines of code to run [SphereNet](https://openreview.net/forum?id=givsRXsOt9r) on [QM9](https://www.nature.com/articles/sdata201422) to incorporate 3D information of molecules.


```python
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

# Load the dataset and split
dataset = QM93D(root='dataset/')
target = 'U0'
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

# Define model, loss, and evaluation
model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                  hidden_channels=128, out_channels=1, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3)                 
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=20, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)

```


1. For details of all included APIs, please refer to the [documentation](https://diveintographs.readthedocs.io/). 
2. We provide a hands-on tutorial for each direction to help you to get started with *DIG*: [Graph Generation](https://diveintographs.readthedocs.io/en/latest/tutorials/graphdf.html), [Self-supervised Learning on Graphs](https://diveintographs.readthedocs.io/en/latest/tutorials/sslgraph.html), [Explainability of Graph Neural Networks](https://diveintographs.readthedocs.io/en/latest/tutorials/subgraphx.html), [Deep Learning on 3D Graphs](https://diveintographs.readthedocs.io/en/latest/tutorials/threedgraph.html), [Graph OOD (GOOD) datasets](https://diveintographs.readthedocs.io/en/latest/tutorials/oodgraph.html).
3. We also provide [examples](https://github.com/divelab/DIG/tree/dig-stable/examples) to use APIs provided in *DIG*. You can get started with your interested directions by clicking the following links.

* [Graph Generation](https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph): [`JT-VAE`](https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/JTVAE), [`GraphAF`](https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphAF), [`GraphDF`](https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphDF), [`GraphEBM`](https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphEBM).
* [Self-supervised Learning on Graphs](https://github.com/divelab/DIG/tree/dig-stable/examples/sslgraph): [`InfoGraph`](https://github.com/divelab/DIG/blob/dig-stable/examples/sslgraph/example_infograph.ipynb), [`GRACE`](https://github.com/divelab/DIG/blob/dig-stable/examples/sslgraph/example_grace.ipynb), [`MVGRL`](https://github.com/divelab/DIG/blob/dig-stable/examples/sslgraph/example_mvgrl.ipynb), [`GraphCL`](https://github.com/divelab/DIG/blob/dig-stable/examples/sslgraph/example_graphcl.ipynb), [`LaGraph (v1 supported)`](https://github.com/divelab/DIG/tree/dig/examples/sslgraph/LaGraph).
* [Explainability of Graph Neural Networks](https://github.com/divelab/DIG/tree/dig-stable/examples/xgraph): [`DeepLIFT`](https://github.com/divelab/DIG/blob/dig-stable/examples/xgraph/deeplift.ipynb), [`GNN-LRP`](https://github.com/divelab/DIG/blob/dig-stable/examples/xgraph/gnn_lrp.ipynb), [`GNNExplainer`](https://github.com/divelab/DIG/blob/dig-stable/examples/xgraph/gnnexplainer.ipynb), [`GradCAM`](https://github.com/divelab/DIG/blob/dig-stable/examples/xgraph/gradcam.ipynb), [`PGExplainer`](https://github.com/divelab/DIG/blob/dig-stable/examples/xgraph/pgexplainer.ipynb), [`SubgraphX`](https://github.com/divelab/DIG/blob/dig-stable/examples/xgraph/subgraphx.ipynb).
* [Deep Learning on 3D Graphs](https://github.com/divelab/DIG/tree/dig-stable/examples/threedgraph): [`SchNet`](https://github.com/divelab/DIG/blob/dig-stable/examples/threedgraph/threedgraph.ipynb), [`DimeNet++`](https://github.com/divelab/DIG/blob/dig-stable/examples/threedgraph/threedgraph.ipynb), [`SphereNet`](https://github.com/divelab/DIG/blob/dig-stable/examples/threedgraph/threedgraph.ipynb).
* [Graph OOD (GOOD) datasets](https://github.com/divelab/DIG/tree/dig-stable/examples/oodgraph): `GOODHIV`, `GOODPCBA`, `GOODZINC`, `GOODCMNIST`, `GOODMotif`, `GOODCora`, `GOODArxiv`, `GOODCBAS`.


## Installation

### Install from pip
The key dependencies of DIG: Dive into Graphs are PyTorch (>=1.10.0), PyTorch Geometric (>=2.0.0), and RDKit.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) (>=1.10.0)

```shell script
$ python -c "import torch; print(torch.__version__)"
>>> 1.10.0
```




2. Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) (>=2.0.0)

```shell script
$ python -c "import torch_geometric; print(torch_geometric.__version__)"
>>> 2.0.0
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


## Contributing

We welcome any forms of contributions, such as reporting bugs and adding new features. Please refer to our [contributing guidelines](https://diveintographs.readthedocs.io/en/latest/contribution/instruction.html) for details.


## Citing DIG

Please cite our [paper](https://jmlr.org/papers/v22/21-0343.html) if you find *DIG* useful in your work:
```
@article{JMLR:v22:21-0343,
  author  = {Meng Liu and Youzhi Luo and Limei Wang and Yaochen Xie and Hao Yuan and Shurui Gui and Haiyang Yu and Zhao Xu and Jingtun Zhang and Yi Liu and Keqiang Yan and Haoran Liu and Cong Fu and Bora M Oztekin and Xuan Zhang and Shuiwang Ji},
  title   = {{DIG}: A Turnkey Library for Diving into Graph Deep Learning Research},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {240},
  pages   = {1-9},
  url     = {http://jmlr.org/papers/v22/21-0343.html}
}
```

## The Team

*DIG: Dive into Graphs* is developed by [DIVE](https://github.com/divelab/)@TAMU. Contributors are Meng Liu*, Youzhi Luo*, Limei Wang*, Yaochen Xie*, Hao Yuan*, Shurui Gui*, Haiyang Yu*, Zhao Xu, Jingtun Zhang, Yi Liu, Keqiang Yan, Haoran Liu, Cong Fu, Bora Oztekin, Xuan Zhang, and Shuiwang Ji.

## Contact

If you have any technical questions, please submit new issues or raise it in our [DIG slack community:fire:](https://join.slack.com/t/dive-into-graphs/shared_invite/zt-1i9kn731c-RhLA1zcEGHXbIToxdVqo0g).

If you have any other questions, please contact us: Meng Liu [mengliu@tamu.edu] and Shuiwang Ji [sji@tamu.edu].


