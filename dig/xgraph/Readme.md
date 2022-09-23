# Explainability of Graph Neural Networks

## Overview

The xgraph package is a collection of benchmark datasets, data interfaces, evaluation metrics, and existing algorithms for explaining GNNs. We aims to provide standardized datasets and unified performance evaluation for academic researchers interested in GNN explanation tasks. 

<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/dig/xgraph/fig/xgraph_new.jpg" width="1000" class="center" alt="logo"/>
    <br/>
</p>


## Implemented Algorithms

The `xgraph` package implements seven existing algorithms for GNN explanation tasks and offers detailed code running instructions. The information about the seven algorithms is summarized in the following table. 

| Method | Links | Brief description |
| ------ | ----- | ------------------ |
| GNNExplainer | [Paper](https://arxiv.org/abs/1903.03894) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/GNNExplainer) | GNNExplainer learns soft masks for edges and hard masks for node features to identify important input information. The masks are randomly initialized and updated to maximize the mutual information between original predictions and new predictions. |
| PGExplainer | [Paper](https://arxiv.org/abs/2011.04573) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/PGExplainer) | PGExplainer learns approximated discrete masks for edges to explain the predictions. It trains a parameterized mask predictor to predict whether an edge is important. All egdes share the same predictor and the preditor is trained to maximize the mutual information. Note that reparameterization trick is used to approximate discrete masks. |
| SubgraphX | [Paper](https://arxiv.org/abs/2102.05152) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/SubgraphX) | SubgraphX explores subgraph-level explanations for deep graph models. It employs the Monte Carlo Tree Search algorithm to efficiently explore different subgraphs via node pruning and select the most important subgraph as the explanation. The importance of subgraphs is measured by Shapley values. |
| GNN-LRP | [Paper](https://arxiv.org/abs/2006.03589) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/GNN-LRP) | GNN-LRP studies the importance of different graph walks via score decomposition. It follows the high-order Taylor decomposition of model predictions to decompose the final predictions to differnet graph walks. In practice, it follows a back propagation procedure to approximate T-order Taylor decomposition termns (T is the number of GNN layers in the model). |
| DeepLIFT | [Paper](https://arxiv.org/abs/1704.02685) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/DeepLIFT) | DeepLIFT is a popular explanation models for image classifer. We extend it to apply for deep graph models. It decomposes the predictions to different nodes abd can be considered as an efficent approximations for Shapley values. |
| Grad-CAM | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/GradCAM) | Grad-CAM is a popular explanation models for image classifer. It is extended to graph models to measure the importance of different nodes. The key idea is to combine the hidden feature maps and the gradients to indicate node importance. |
| XGNN | [Paper](https://arxiv.org/abs/2006.02587) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/XGNN) | XGNN is a model-level explanation method for Graph Neural Networks. Instead of explaining specific predictions, it studies the general behavior of GNNs, such as what input graph patterns will maximize the prediction of a certai class. It employs graph generation algorithm to generate graphs to maximize a target prediction score. |
| TAGE | [Paper](https://arxiv.org/abs/2202.08335) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/TAGE) | Task-Agnostic GNN Explainer (TAGE) is proposed to fulfill the desire of explaining two-stage trained models and multitask models. The explainer is independent of downstream models and trained under self-supervision with no knowledge of downstream tasks. TAGE enables the explanation of GNN embedding models with unseen downstream tasks and allows efficient explanation of multitask models. |

## Package Usage

Here we briefly introduce our unified data interfaces and evaluation metrics.

(1) Data interfaces

We provide unified data interfaces for reading several benchmark datasets and a standard Pytorch data loader. The details can be found in the `datasets` folder ([link](https://github.com/divelab/DIG/tree/main/dig/xgraph/datasets)).


(2) Evaluation metrics

We also provide unified evaluation metrics, including Fidelity+, Fidelity-, and Sparsity, for the evaluation of explanation results. We provide demo code to help the usage of these metrics. The details can be found in `metrics` folder ([link](https://github.com/divelab/DIG/tree/main/dig/xgraph/metrics)).

## Citations
If you use our code and data, please cite our papers.

```
@article{yuan2020explainability,
  title={Explainability in Graph Neural Networks: A Taxonomic Survey},
  author={Yuan, Hao and Yu, Haiyang and Gui, Shurui and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.15445},
  year={2020}
}
```
## Contact
*If you have any questions, please submit an issue or contact us at Hao Yuan [hao.yuan@tamu.edu] and Shuiwang Ji [sji@tamu.edu] .*
