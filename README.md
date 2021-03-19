<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>






[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg
[license-url]:https://github.com/divelab/DIG/blob/main/LICENSE
[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]:https://github.com/divelab/DIG/blob/main/CONTRIBUTING.md

![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG&)
![Contributors](https://img.shields.io/github/contributors/divelab/DIG)
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]

## Overview
*DIG: Dive into Graphs* is a turnkey library for graph deep learning research. [[Paper (Coming soon)]](https://github.com/divelab/DIG)

It includes unified implementations of **data interfaces**, **common algorithms**, and **evaluation metrics** for several advanced tasks. Our goal is to enable researchers to easily implement and benchmark algorithms. Currently, we consider the following research directions.

* [Graph Generation](https://github.com/divelab/DIG/tree/main/dig/ggraph)
* [Self-supervised Learning on Graphs]()
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
* Self-supervised Learning on Graphs: [`sslgraph`]()
* Explainability of Graph Neural Networks: [`xgraph`](https://github.com/divelab/DIG/tree/main/dig/xgraph)
* Deep Learning on 3D Graphs: [`3dgraph`](https://github.com/divelab/DIG/tree/main/dig/3dgraph)


## Contributing

We welcome any forms of contributions, such as reporting bugs and adding new algorithms. Please refer to our [contributing guidelines](https://github.com/divelab/DIG/blob/main/CONTRIBUTING.md) for details.


## Citing DIG

Please cite our paper if you find *DIG* useful in your work:
```
To be added
```
