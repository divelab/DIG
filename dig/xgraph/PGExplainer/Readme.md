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
$ source  ./install.sh
```

## Usage
* Download the required [dataset](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ET69UPOa9jxAlob03sWzJ50BeXM-lMjoKh52h6aFc8E8Jw?e=lglJcP) to `DIG/xgraph/dataset`
* Download the [checkpoints](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/EYydmjDKl7xPsqdRaJc-se4BZSea6EI53dMlZHoM9fTvdg?e=I42r6H) to  `DIG/xgraph/PGExplainer/checkpoint`
* run the pipeline scripts with corresponding dataset
```shell script
$ source ./scripts.sh
```

Moreover, if you want to train a new model for these datasets, 
run the training scripts for corresponding dataset.
```shell script
$ cd DIG/xgraph/PGExplainer
$ source ./models/train_gnns.sh 
```


## Citations
``` 
@article{yuan2020explainability,
  title={Explainability in Graph Neural Networks: A Taxonomic Survey},
  author={Yuan, Hao and Yu, Haiyang and Gui, Shurui and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.15445},
  year={2020}
}
```
