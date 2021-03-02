# PGExplainer

This repository is a PyTorch and 
PyTorch Geometric implementation of Paper 
[Parameterized Explainer for Graph Neural Network](https://arxiv.org/abs/2011.04573).


The [official implementation](https://github.com/flyingdoog/PGExplainer) is based on TensorFlow.

## Installation
* clone this repository
* create the required environment
```shell script
$ git clone https://github.com/divelab/DIG.git
$ cd DIG/xgraph/PGexplainer
$ source  ./install.bash
```

## Usage
* Download the required dataset to `DIG/xgraph/dataset`
* Download the checkpoints to  `DIG/xgraph/PGExplainer/checkpoint`
* run the pipeline scripts with corresponding dataset
```shell script
$ sh scripts.sh
```

## Citations
``` 
@article{luo2020parameterized,
  title={Parameterized Explainer for Graph Neural Network},
  author={Luo, Dongsheng and Cheng, Wei and Xu, Dongkuan and Yu, Wenchao and Zong, Bo and Chen, Haifeng and Zhang, Xiang},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```