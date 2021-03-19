<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/imgs/DIG-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>

------



## Overview
*DIG: Dive into Graphs* is a turnkey library for graph deep learning research. [[Paper (Coming soon)]](https://github.com/divelab/DIG)

It includes unified implementations of **data interfaces**, **common algorithms**, and **evaluation metrics** for several advanced tasks. Our goal is to enable researchers to easily implement and benchmark algorithms. Currently, we consider the following research directions.

* [Graph Generation](https://github.com/divelab/DIG/tree/main/dig/ggraph)
* [Self-supervised Learning on Graphs]()
* [Explainability of Graph Neural Networks](https://github.com/divelab/DIG/tree/main/dig/xgraph)
* [Deep Learning on 3D Graphs](https://github.com/divelab/DIG/tree/main/dig/3dgraph)

(To do: Add the overview figure of the library.)

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
