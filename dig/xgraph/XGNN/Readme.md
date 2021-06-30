# XGNN

This repository provides the code for our paper [XGNN: Towards Model-Level Explanations of Graph Neural Networks](https://arxiv.org/abs/2006.02587) accepted by KDD-2020. 

Our proposed XGNN algorithm aims at providing model-level explanations for Graph Neural Networks. To the best of our knowledge, this is the first attempt to study the model-level explanations of Graph Neural Networks. In this repository, we show how to explain a GCN classifier trained on the MUTAG dataset. 


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

Place the checkpoint of the GNNs to be explained in the checkpoint folder. Also, modify corresponding paths if needed.

In "gnn.py" we provide an example showing the training of GNNs, and then the trained GNNs become the model to be explained. 

Our data and checkpoint are available (https://drive.google.com/drive/u/2/folders/1To5IQa-3H_m48OwhJzEhIwz1swnHcOoz). 

## The XGNN Algorithm

The policy network of our XGNN is defined in "policy_nn.py". You can modify it as needed, depending on the GNNs/tasks at hand. 

The explanation generation stage is defined in "gnn_explain.py". You can tune the hyper-parameters as needed. 

Simply call "main.py" to obtain explanations after proper settings and modifications. 

After training, the generated explanations should maximize the predictions of a certain class (or other targets). We found that there are multiple graph patterns that can maximize the predicted probability of a target class. 


## How to customize?

Our XGNN is a general framework, you can customize it for your own task. 

- Define your data/graph properties as needed. In this repository, we show how to explain a GCN classifier trained on the MUTAG dataset so each node is representing an atom. 

- Define your own graph rules in "gnn_explain.py". In our example, the check_validity function checks whether the generated graph is valid. 

- You can customize the roll_out function in "gnn_explain.py". For simple tasks on synthetic data, roll_out is not necessary. In addition, there are several ways to handle invalid generated graphs in the roll_out. In this example, we simply return a negative pre-defined reward. 

- The GNN layer, policy network architectures, and normalize_adj functions can be easily replaced by any suitable functions. 

- Our provided code is based on CPU so you can monitor the explanation generation step by step with IDEs, such as Spyder. 

- You can customize the target of explanations. Currently, our code explains the predictions of different classes. You may modify this to study what explanations activate other network targets, such as hidden neurons/ channels. 
