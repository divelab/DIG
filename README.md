<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>






[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg
[license-url]:https://github.com/divelab/DIG/blob/main/LICENSE
[contributor-image]:https://img.shields.io/github/contributors/divelab/DIG
[contributor-url]:https://github.com/divelab/DIG/graphs/contributors
[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]:https://github.com/divelab/DIG/blob/main/CONTRIBUTING.md


![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![Contributors][contributor-image]][contributor-url]
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]


*DIG: Dive into Graphs* is a turnkey library for graph deep learning research.

[DIG: A Turnkey Library for Diving into Graph Deep Learning Research](https://arxiv.org/abs/2103.12608)

Meng Liu*, Youzhi Luo*, Limei Wang*, Yaochen Xie*, Hao Yuan*, Shurui Gui, Zhao Xu, Haiyang Yu, Jingtun Zhang, Yi Liu, Keqiang Yan, Bora Oztekin, Haoran Liu, Xuan Zhang, Cong Fu, and Shuiwang Ji.

## Why DIG?

The key difference with current graph deep learning libraries, such as PyTorch Geometric (PyG) and Deep Graph Library (DGL), is that, while PyG and DGL support basic graph deep learning operations, DIG provides a unified testbed for higher level, research-oriented graph deep learning tasks, such as graph generation, self-supervised learning, explainability, and 3D graphs.

If you are working or plan to work on research in graph deep learning, DIG enables you to develop your own methods within our extensible framework, and compare with current baseline methods using common datasets and evaluation metrics without extra efforts.

## Overview

It includes unified implementations of **data interfaces**, **common algorithms**, and **evaluation metrics** for several advanced tasks. Our goal is to enable researchers to easily implement and benchmark algorithms. Currently, we consider the following research directions.

* [Graph Generation](https://github.com/divelab/DIG/tree/main/dig/ggraph)
* [Self-supervised Learning on Graphs](https://github.com/divelab/DIG/tree/main/dig/sslgraph)
* [Explainability of Graph Neural Networks](https://github.com/divelab/DIG/tree/main/dig/xgraph)
* [Deep Learning on 3D Graphs](https://github.com/divelab/DIG/tree/main/dig/3dgraph)

<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-overview.jpg" width="700" class="center" alt="logo"/>
    <br/>
</p>

## Usage

*DIG* could be large since it consists of multiple research directions. You are recommended to download the subdirectory of interest. For example, if you are interested in graph generation (`ggraph`):

```shell script
svn export https://github.com/divelab/DIG/trunk/ggraph
```

We provide documentations on how to use our implemented data interfaces and evaluation metrics. For every algorithm, an instruction documentation is also available. You can get started with your interested directions.

* Graph Generation: [`ggraph`](https://github.com/divelab/DIG/tree/main/dig/ggraph)
* Self-supervised Learning on Graphs: [`sslgraph`](https://github.com/divelab/DIG/tree/main/dig/sslgraph)
* Explainability of Graph Neural Networks: [`xgraph`](https://github.com/divelab/DIG/tree/main/dig/xgraph)
* Deep Learning on 3D Graphs: [`3dgraph`](https://github.com/divelab/DIG/tree/main/dig/3dgraph)


## Contributing

We welcome any forms of contributions, such as reporting bugs and adding new algorithms. Please refer to our [contributing guidelines](https://github.com/divelab/DIG/blob/main/CONTRIBUTING.md) for details.


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

## Contact

If you have any technical questions, please submit a new issue.

If you have other questions, please contact us: Meng Liu [mengliu@tamu.edu] or Shuiwang Ji [sji@tamu.edu].


