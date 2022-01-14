# Explainability of Graph Neural Networks

## Overview

Benchmark for xgraph provides a systematic schema to compare different instance-wise explanation algorithms for GNNs.
With the same datasets and evaluation metrics, we can measure the importance scores for the explanations provided by algorithms. 

<p align="center">
<img src="imgs/xgraph.jpg" width="1000" class="center" alt="logo"/>
    <br/>
</p>


## Implemented Algorithms

Now, the xgraph benchmark provides unified comparison for seven existing algorithms. 
The information about the seven algorithms is summarized in the following table. 

| Method | Links | Brief description |
| ------ | ----- | ------------------ |
| GNNExplainer | [Paper](https://arxiv.org/abs/1903.03894) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/GNNExplainer) | GNNExplainer learns soft masks for edges and hard masks for node features to identify important input information. The masks are randomly initialized and updated to maximize the mutual information between original predictions and new predictions. |
| PGExplainer | [Paper](https://arxiv.org/abs/2011.04573) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/PGExplainer) | PGExplainer learns approximated discrete masks for edges to explain the predictions. It trains a parameterized mask predictor to predict whether an edge is important. All egdes share the same predictor and the preditor is trained to maximize the mutual information. Note that reparameterization trick is used to approximate discrete masks. |
| SubgraphX | [Paper](https://arxiv.org/abs/2102.05152) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/SubgraphX) | SubgraphX explores subgraph-level explanations for deep graph models. It employs the Monte Carlo Tree Search algorithm to efficiently explore different subgraphs via node pruning and select the most important subgraph as the explanation. The importance of subgraphs is measured by Shapley values. |
| GNN-LRP | [Paper](https://arxiv.org/abs/2006.03589) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/GNN-LRP) | GNN-LRP studies the importance of different graph walks via score decomposition. It follows the high-order Taylor decomposition of model predictions to decompose the final predictions to differnet graph walks. In practice, it follows a back propagation procedure to approximate T-order Taylor decomposition termns (T is the number of GNN layers in the model). |
| DeepLIFT | [Paper](https://arxiv.org/abs/1704.02685) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/DeepLIFT) | DeepLIFT is a popular explanation models for image classifer. We extend it to apply for deep graph models. It decomposes the predictions to different nodes abd can be considered as an efficent approximations for Shapley values. |
| Grad-CAM | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/GradCAM) | Grad-CAM is a popular explanation models for image classifer. It is extended to graph models to measure the importance of different nodes. The key idea is to combine the hidden feature maps and the gradients to indicate node importance. |
| XGNN | [Paper](https://arxiv.org/abs/2006.02587) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/XGNN) | XGNN is a model-level explanation method for Graph Neural Networks. Instead of explaining specific predictions, it studies the general behavior of GNNs, such as what input graph patterns will maximize the prediction of a certai class. It employs graph generation algorithm to generate graphs to maximize a target prediction score. |

Besides, we also provide the RandomExplainer which gives random explanations for the input graphs.



## Results
We also provide the experiment results for GCN model on six different datasets. 

We provide the checkpoints [here](https://drive.google.com/drive/folders/19krHmYpGDmR4abDB3bz0fVLYUeFmsmR7?usp=sharing) 
and part of the results [here](https://drive.google.com/drive/folders/1zNm9i1XvAMeZsmvzS1fyeIwpnaU8AkK7?usp=sharing).


## Citations
If you use our code and data, please cite our survey paper and DIG paper.

```
@article{yuan2020explainability,
  title={Explainability in Graph Neural Networks: A Taxonomic Survey},
  author={Yuan, Hao and Yu, Haiyang and Gui, Shurui and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.15445},
  year={2020}
}
```

```
@article{JMLR:v22:21-0343,
  author  = {Meng Liu and Youzhi Luo and Limei Wang and Yaochen Xie and Hao Yuan and Shurui Gui and Haiyang Yu and Zhao Xu and Jingtun Zhang and Yi Liu and Keqiang Yan and Haoran Liu and Cong Fu and Bora M Oztekin and Xuan Zhang and Shuiwang Ji},
  title   = {{DIG}: A Turnkey Library for Diving into Graph Deep Learning Research},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {240},
  pages   = {1-9},
  url     = {http://jmlr.org/papers/v22/21-0343.html}
}
```
