# XGNN

This repository provides the code for our paper [XGNN: Towards Model-Level Explanations of Graph Neural Networks](https://arxiv.org/abs/2006.02587) accepted by KDD-2020. 

Our proposed XGNN algorithm aims at providing model-level explanations for Graph Neural Networks. To the best of our knowledge, this is the first attempt to study the model-level explanations of Graph Neural Networks. 


# Citations
If using this code , please cite our paper.
```
@inproceedings{yuan2020xgnn,
  title={Xgnn: Towards model-level explanations of graph neural networks},
  author={Yuan, Hao and Tang, Jiliang and Hu, Xia and Ji, Shuiwang},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={430--438},
  year={2020}
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

# Usage

## Dataset and Checkpoint

Download the dataset and modify corresponding paths. 

Place the checkpoint of the GNNs to be explained in checkpoint folder. Also modify corresponding paths if needed.

In “gnn.py” we provide an example showing the training of GNNs, and then the trained GNNs become the target of explanations. 

Our data and checkpoint are available upon request. 

## XGNN Algorithm




