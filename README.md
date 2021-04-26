<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>






[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg
[license-url]:https://github.com/divelab/DIG/blob/main/LICENSE
[contributor-image]:https://img.shields.io/github/contributors/divelab/DIG
[contributor-url]:https://github.com/divelab/DIG/graphs/contributors
[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]:https://diveintographs.readthedocs.io/en/latest/intro/introduction.html


![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![Contributors][contributor-image]][contributor-url]
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]


**[Documentation](https://diveintographs.readthedocs.io)** | **[Paper](https://arxiv.org/abs/2103.12608)**| **[Benchmarks/Examples](https://github.com/divelab/DIG/tree/dig/benchmarks)**

*DIG: Dive into Graphs* is a turnkey library for graph deep learning research.


## Why DIG?

The key difference with current graph deep learning libraries, such as PyTorch Geometric (PyG) and Deep Graph Library (DGL), is that, while PyG and DGL support basic graph deep learning operations, DIG provides a unified testbed for higher level, research-oriented graph deep learning tasks, such as graph generation, self-supervised learning, explainability, and 3D graphs.

If you are working or plan to work on research in graph deep learning, DIG enables you to develop your own methods within our extensible framework, and compare with current baseline methods using common datasets and evaluation metrics without extra efforts.

## Overview

It includes unified implementations of **data interfaces**, **common algorithms**, and **evaluation metrics** for several advanced tasks. Our goal is to enable researchers to easily implement and benchmark algorithms. Currently, we consider the following research directions.

* **Graph Generation**
* **Self-supervised Learning on Graphs**
* **Explainability of Graph Neural Networks**
* **Deep Learning on 3D Graphs**



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

3. Install [RDKit](https://github.com/rdkit/rdkit).

```shell script
conda install -y -c conda-forge rdkit
```
    
4. Install DIG: Dive into Graphs.

```shell script
pip install dive-into-graphs
```

### Install from source
If you want to try the latest features that have not been released yet, you can install dig from source.

```shell script
git clone https://github.com/divelab/DIG.git
cd DIG
python setup.py install
```


## Usage

We provide [benchmark implementations](https://github.com/divelab/DIG/tree/dig/benchmarks) as examples to use *DIG*. For details of all included APIs, please refer to the [documentation](https://diveintographs.readthedocs.io/).

## Contributing

We welcome any forms of contributions, such as reporting bugs and adding new algorithms. Please refer to our [contributing guidelines](https://diveintographs.readthedocs.io/en/latest/contribution/instruction.html) for details.


## Citing DIG

Please cite our paper if you find *DIG* useful in your work:
```
@article{liu2021dig,
      title={{DIG}: A Turnkey Library for Diving into Graph Deep Learning Research}, 
      author={Meng Liu and Youzhi Luo and Limei Wang and Yaochen Xie and Hao Yuan and Shurui Gui and Zhao Xu and Haiyang Yu and Jingtun Zhang and Yi Liu and Keqiang Yan and Bora Oztekin and Haoran Liu and Xuan Zhang and Cong Fu and Shuiwang Ji},
      journal={arXiv preprint arXiv:2103.12608},
      year={2021},
}
```

## The Team

*DIG: Dive into Graphs* is developed by [DIVE](https://github.com/divelab/)@TAMU. Contributors are Meng Liu*, Youzhi Luo*, Limei Wang*, Yaochen Xie*, Hao Yuan*, Shurui Gui, Zhao Xu, Haiyang Yu, Jingtun Zhang, Yi Liu, Keqiang Yan, Bora Oztekin, Haoran Liu, Xuan Zhang, Cong Fu, and Shuiwang Ji.

## Contact

If you have any technical questions, please submit new issues.

If you have any other questions, please contact us: Meng Liu [mengliu@tamu.edu] or Shuiwang Ji [sji@tamu.edu].


