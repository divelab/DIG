# SubgraphX

This is the official implement of Paper 
[On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/abs/2102.05152)


## Installation
* clone the repository 
* create the env and install the requirements

```shell script
$ git clone https://github.com/divelab/DIG.git
$ cd DIG/xgraph/SubgraphX
$ source ./install.sh
```

## Usage
* Download the required [dataset](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ET69UPOa9jxAlob03sWzJ50BeXM-lMjoKh52h6aFc8E8Jw?e=lglJcP) to `DIG/xgraph/dataset`
* Download the [checkpoints](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/EYydmjDKl7xPsqdRaJc-se4BZSea6EI53dMlZHoM9fTvdg?e=I42r6H) to `DIG/xgraph/SubgraphX/checkpoint`
* run the searching scripts with corresponding dataset
```shell script
$ cd DIG/xgraph/SubgraphX
$ source ./scripts.sh
``` 
The hyper-parameters for different models and datasets are shown in this script.

In addition, we also provide the saved searching results.
If you want to reproduce, you can directly download the 
[results](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ERxIONDcl8xKswisrsbHo2MBoEwPAjFruUzwsLpESwalxA?e=IuFanz)
 to `DIG/xgraph/SubgraphX/results`

Moreover, if you want to train a new model for these datasets, 
run the training scripts for corresponding dataset.
```shell script
$ cd DIG/xgraph/SubgraphX
$ source ./models/train_gnns.sh 
```

## Citations
If you use this code, please cite our papers.

```
@misc{yuan2021explainability,
      title={On Explainability of Graph Neural Networks via Subgraph Explorations}, 
      author={Hao Yuan and Haiyang Yu and Jie Wang and Kang Li and Shuiwang Ji},
      year={2021},
      eprint={2102.05152},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@article{yuan2020explainability,
  title={Explainability in Graph Neural Networks: A Taxonomic Survey},
  author={Yuan, Hao and Yu, Haiyang and Gui, Shurui and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.15445},
  year={2020}
}
```