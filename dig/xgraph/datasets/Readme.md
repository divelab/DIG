# Datasets

The datasets part gives a list of all the applied graph datasets in the xgraph package.

More details of these datasets are described in [our paper](https://arxiv.org/abs/2012.15445).

In addion, we provide a unified data reader so different types of data can be easily loaded. The data reader supports all following datasets and the details can be found in `load_datasets.py` file.
  
## Synthetic datasets
The synthetic datasets are can be downloaded from [here](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ETK3CTHgFQVFhzIFB5piofQBv2Av-TsX_rnweymigve_hg?e=SyQjEX)

#### Node classification 
* BA-shapes 
* BA-Community
* Tree-Cycle 
* Tree-Grids

#### Graph classification 
* BA-2Motifs 
* BA-LRP 

## Text2Graph datasets (sentiment graph data)

Text data consists of words and phrases with human-understandable semantic meanings, and the results of explainability are easy to be understood.
However, traditional text data is usually sequence data and can't be applied to graph neural networks directly.
Therefore, we provide a way to transfer the traditional text classification datasets to graph structure datasets.
For each sentence, we take tokens as nodes and dependency relationships as edges to construct graph data.
In this way, we transfer the sentiment classification datasets SST2, SST5 and Twitter datasets into graph classification datasets, 
and we call them sentiment graph data.

The corresponding Graph-SST2, Graph-SST5 and Graph-Twitter can be downloaded
[here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/yhy12138_mail_ustc_edu_cn/EsEDizdNoudJvJibNXK0ImwB5iDVJhCeiycFxnTDigtjow?e=qoaLgy).

* Graph-SST2
* Graph-SST5
* Graph-Twitter 


### Molecule datasets

For the former four molecule datasets, we directly use the corresponding classes from 
[Moleculenet](http://moleculenet.ai/datasets-1) 
in Pytorch-Geometric, and they can be downloaded automatically. 
In addition, we provide the download link for MUTAG datasets.

* BBBP
* Tox21
* BACE
* ClinTox
* [MUTAG](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/Ee52X4fj-KBDrW4T2LJ_XlYBXjgKru_miEbu9rbN26Tdtg?e=gsyBk5)


## Usage

We provide a unified data loader interface in `load_datasets.py`.
In the following demo, the `get_datasets` function can load all the applied datasets and return a class of `torch_geometric.data.InMemoryDataset`. 

```python
# for multi-task datasets, such as Tox21 and BACE, the target task should be specified.
from load_datasets import get_dataset
dataset = get_dataset(dataset_dir='./datasets', dataset_name=bbbp, task=None)
```

For graph classification datasets, we provide an interface `get_dataloader` to support the subsequent batch processing process, 
and split the dataset into training, deving, and testing datsets.

```python
from load_datasets import get_dataset, get_dataloader
data_loader = get_dataloader(dataset,                           # data_loader: dict, following the structure {'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset}
                             batch_size=32,                     # batch_size: int
                             random_split_flag=True,            # random_split_flag：bool, True when randomly split the dataset into training, deving and testing datasets.
                             data_split_ratio=[0.8, 0.1, 0.1],  # data_split_ratio: list, the ratio of data in training, deving and testing datasets.
                             seed=2)                            # seed: int, random seed for randomly split the dataset
```

## Citations
If you use our code and data, please cite our papers.

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
