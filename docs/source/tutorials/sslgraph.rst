=======================================
Tutorial for Self-Supervised GNNs
=======================================

In this tutorial, we will show how to use DIG library [1]_ to build self-supervised learning (SSL) 
frameworks to train Graph Neural Networks (GNNs). Specifically, we provide a unified and general 
description of the constrastive SSL framework following the survey [2]_, based on which we introduce 
the :class:`~dig.sslgraph.method.Contrastive` base class. We further provide two examples to demonstrate 
how to build up customized contrastive framework and the GraphCL [3]_ framework using the base class.

.. contents::
    :local:

Contrastive Learning Frameworks
-------------------------------

Contrastive learning has achieved great success in natural language processing and computer vision. 
Inspired by the success, contrastive learning methods are adapted to graph data, such as GraphCL, 
MVGRL [5]_ and InfoGraph [6]_ . 

A contrastive learning framework consists of multiple (k>=2) views of the input variable. It seeks to 
maximize the mutual information between the representations of different views. In particular, it learns 
to discriminate jointly sampled view pairs from independently sampled view pairs. Concretely, two views 
generated from the same instance are considered as a positive pair and two views generated from different 
instances are considered as a negative pair.

The computation pipeline of a contrastive learning framework can be formulated as

.. math::

    \mathbf{w_i} = \mathcal{T}_i (\mathbf{A}, \mathbf{X})

.. math::

    \mathbf{h_i} = f_i (\mathbf{w_i}), i = 1, \cdots, k

.. math::
    
    \max_{\{f_i\}_{i=1}^k} \frac{1}{\sum_{i \ne j}\sigma_{ij} } \left[ 
    \sum_{i\ne j} \sigma_{ij} \mathcal{I}(\mathbf{h_i}, \mathbf{h_j}) \right]

where 
:math:`(\mathbf{A}, \mathbf{X})` is a given graph as a random variable distributed from :math:`\mathcal{P}`,
:math:`\mathcal{T}_1,\cdots,\mathcal{T}_k` are multiple transformations (augmentations) to obtain different views 
:math:`\mathbf{w_1}, \cdots, \mathbf{w_k}`, :math:`f_i, \cdots, f_k` are encoding networks to generate output
representations :math:`\mathbf{h_1}, \cdots, \mathbf{h_k}`, :math:`\sigma_{ij} \in \{0,1\}`, :math:`\mathcal{I}(\mathbf{h_i}, \mathbf{h_j})` is mutual information between :math:`\mathbf{h_i}` and 
:math:`\mathbf{h_j}`, :math:`\sigma_{ij}=1`
if the mutual information is computed between :math:`\mathbf{h_i}` and :math:`\mathbf{h_j}`, and :math:`\sigma_{ij}=0` 
otherwise. 


The "Contrastive" Base Class
----------------------------

DIG-sslgraph provides the :class:`~dig.sslgraph.Contrastive` base class to implement existing or create customized 
contrastive learning frameworks.

**Contrastive(objective, views_fn, graph_level=True, node_level=False, z_dim=None, z_n_dim=None, proj=None, proj_n=None, neg_by_crpt=False, tau=0.5, device=None, choice_model='last',model_path='models')**

Following the computation pipeline, to construct a certain contrastive framework, one need to first specify key 
components in the :class:`~dig.sslgraph.method.Contrastive` base class: 1) :obj:`objective`, the objective function 
for MI maximization, 2) :obj:`views_fn`, a list of transformations to generate multiple views of a graph, and 3) 
:obj:`graph_level` and :obj:`node_level`, whether to learn graph-level or node-level representations.


Objectives (MI Estimators)
++++++++++++++++++++++++++

Two lower-bound mutual information estimators are implemented in DIG:

.. Donsker-Varadhan Estimator (DV)
.. #############################

- Jensen-Shannon Estimator (``JSE``):

.. math::

    \hat{\mathcal{I}}^{(JS)}(\mathbf{h_i}, \mathbf{h_j}) = 
    \mathbb{E}_{(\mathbf{A}, \mathbf{X}) \sim \mathcal{P}} \left[ log(\mathcal{D}(\mathbf{h_i}, \mathbf{h_j})) \right] +
    \mathbb{E}_{[(\mathbf{A}, \mathbf{X}), (\mathbf{A'}, \mathbf{X'})] \sim \mathcal{P} \times \mathcal{P}}
    \left[ log(1-\mathcal{D}(\mathbf{h_i}, \mathbf{h_j'})) \right]

where :math:`\mathbf{h_i}, \mathbf{h_j}` in the first term are computed from :math:`(\mathbf{A}, \mathbf{X})`
distributed from :math:`\mathcal{P}`, :math:`\mathbf{h_i}` and :math:`\mathbf{h_j}'` in the second term are 
computed from :math:`(\mathbf{A}, \mathbf{X})` and :math:`(\mathbf{A'}, \mathbf{X'})` identically and independently
distributed from the distribution :math:`\mathcal{P}`.

- InfoNCE (``NCE``):

.. math::

    \hat{\mathcal{I}}^{(NCE)}(\mathbf{h_i}, \mathbf{h_j}) &= 
    \mathbb{E}_{(\mathbf{A}, \mathbf{X}) \sim \mathcal{P}} \left[ \mathcal{D}(\mathbf{h_i}, \mathbf{h_j}) -
        \mathbb{E}_{\mathbf{K}\sim \mathcal{P}^N} \left[ log \sum_{(\mathbf{A'}, \mathbf{X'}) \in \mathbf{K}} 
        e^{\mathcal{D}(\mathbf{h_i}, \mathbf{h_j}') / N} \left| \right (\mathbf{A}, \mathbf{X}) \right] \right] \\
    &= \mathbb{E}_{[(\mathbf{A}, \mathbf{X}), \mathbf{K}] \sim \mathcal{P} \times \mathcal{P}^N} \left[
        log \frac{e^{(\mathbf{h_i}, \mathbf{h_j})}}{\sum_{(\mathbf{A'}, \mathbf{X'}) \in \mathbf{K}}
        e^{\mathcal{D}(\mathbf{h_i}, \mathbf{h_j}')}}\right] + logN

where :math:`\mathbf{K}` consisted of :math:`N` random variable identically and independently distributed from
:math:`\mathcal{P}`, :math:`\mathbf{h_i}, \mathbf{h_j}` are the representations of the `i`-th and `j`-th views
of :math:`(\mathbf{A}, \mathbf{X})`, and :math:`\mathbf{h_i}'` is the representation of the `j`-th view of 
:math:`(\mathbf{A'}, \mathbf{X'})`.


- In addition to the type of MI estimator, the users are able to specify :obj:`proj` and :obj:`proj_n` whether projection head(s) with trainable parameters are included and what projection head(s) to be included when computing the MI estimates. A projection head will turn the MI estimator into a parameterized estimator and can bring performance gain to certain contrastive methods.



View Generation
+++++++++++++++

Variety of view generation functions :math:`\mathcal{T}` belonging to three types are implemented in DIG. To perform
multi-view contrastive learning the number of view generators (:obj:`len(views_fn)`) should be no less than 2.

- Feature transformations (:class:`~dig.sslgraph.method.contrastive.views_fn.NodeAttrMask`):

.. math::

    \mathcal{T}_{feat}(\mathbf{A}, \mathbf{X}) = (\mathbf{A}, \mathcal{T}_X(\mathbf{X}))

where :math:`\mathcal{T}_X: \mathbb{R}^{|V|\times d} \to \mathbb{R}^{|V|\times d}` performs the 
transformation on the feature matrix :math:`\mathbf{X}`.


- Structure transformations (:class:`~dig.sslgraph.method.contrastive.views_fn.EdgePerturbation`, :class:`~dig.sslgraph.method.contrastive.views_fn.Diffusion`, :class:`~dig.sslgraph.method.contrastive.views_fn.DiffusionWithSample`)

.. math::

    \mathcal{T}_{struct}(\mathbf{A}, \mathbf{X}) = (\mathcal{T}_A(\mathbf{A}), \mathbf{X})

where :math:`\mathcal{T}_A: \mathbb{R}^{|V|\times |V|} \to \mathbb{R}^{|V|\times |V|}` performs the 
transformation on the adjacency matrix :math:`\mathbf{A}`.


- Sampling-based transformations (:class:`~dig.sslgraph.method.contrastive.views_fn.UniformSample`, :class:`~dig.sslgraph.method.contrastive.views_fn.RWSample`)

.. math::

    \mathcal{T}_{sample}(\mathbf{A}, \mathbf{X}) = (\mathbf{A}[S;S], \mathbf{X}[S])

where :math:`S \subseteq V` denotes a subset of nodes and :math:`[\cdot]` selects certain rows and
columns from a matrix based on indices of nodes in :math:`S`.


- We further provide :class:`~dig.sslgraph.method.contrastive.views_fn.Sequential` and :class:`~dig.sslgraph.method.contrastive.views_fn.RandomView` to generate views based on multiple transformations.


Level of Representations
++++++++++++++++++++++++
DIG-sslgraph provides three different representation levels to perform contrastive learning. By default, the base class
performs graph-level contrast. To perform node-level contrast, one can set :obj:`graph_level`=:bool:`False` and 
:obj:`node_level`=:bool:`True`. If both :obj:`graph_level` and :obj:`node_level` are :obj:`True`, the contrastive method
performs local-global constrast. In this case, the number of view generators (:obj:`len(views_fn)`) can be 1.



Creating Customized Contrastive Methods
---------------------------------------

The simplest way to create a customized contrastive method is to define a subclass of :class:`~dig.sslgraph.Contrastive` by
specify corresponding components and override the method :obj:`train()`. Below is an example to employ two node attribute masking
view functions, the "JSE" objective with "MLP" projection head for graph-level constrastive learning.

.. code-block ::
    
    from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask
    from dig.sslgraph.method import Contrastive

    class SSLModel(Contrastive):
        def __init__(self, z_dim, mask_ratio, **kwargs):
        
            objective = "JSE"
            proj="MLP"
            mask_i = NodeAttrMask(mask_ratio=mask_ratio)
            mask_j = NodeAttrMask(mask_ratio=mask_ratio)
            views_fn = [mask_i, mask_j]
        
            super(SSLModel, self).__init__(objective=objective,
                                        views_fn=views_fn,
                                        z_dim=z_dim,
                                        proj=proj,
                                        node_level=False,
                                        **kwargs)
                                        
        def train(self, encoder, data_loader, optimizer, epochs, per_epoch_out=False):
            for enc, proj in super(SSLModel, self).train(encoder, data_loader, 
                                                        optimizer, epochs, per_epoch_out):
                yield enc

    ssl_model = SSLModel(z_dim=embed_dim, mask_ratio=0.1)


Below is another example using the :class:`~dig.sslgraph.Contrastive` base class to implement GraphCL, who employs
random augmentations to generate views and optimize the "NCE" objective.


.. code-block ::
    
    import sys, torch
    import torch.nn as nn
    from dig.sslgraph.method import Contrastive
    from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
        UniformSample, RWSample, RandomView

    class GraphCL(Contrastive):

        def __init__(self, dim, aug_1=None, aug_2=None, aug_ratio=0.2, **kwargs):

            views_fn = []

            for aug in [aug_1, aug_2]:
                if aug is None:
                    views_fn.append(lambda x: x)
                elif aug == 'dropN':
                    views_fn.append(UniformSample(ratio=aug_ratio))
                elif aug == 'permE':
                    views_fn.append(EdgePerturbation(ratio=aug_ratio))
                elif aug == 'subgraph':
                    views_fn.append(RWSample(ratio=aug_ratio))
                elif aug == 'maskN':
                    views_fn.append(NodeAttrMask(mask_ratio=aug_ratio))
                elif aug == 'random2':
                    canditates = [UniformSample(ratio=aug_ratio),
                                  RWSample(ratio=aug_ratio)]
                    views_fn.append(RandomView(canditates))
                elif aug == 'random4':
                    canditates = [UniformSample(ratio=aug_ratio),
                                  RWSample(ratio=aug_ratio),
                                  EdgePerturbation(ratio=aug_ratio)]
                    views_fn.append(RandomView(canditates))
                elif aug == 'random3':
                    canditates = [UniformSample(ratio=aug_ratio),
                                  RWSample(ratio=aug_ratio),
                                  EdgePerturbation(ratio=aug_ratio),
                                  NodeAttrMask(mask_ratio=aug_ratio)]
                    views_fn.append(RandomView(canditates))
                else:
                    raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                    'maskN', 'random2', 'random3', 'random4'] or None.")

            super(GraphCL, self).__init__(objective='NCE',
                                          views_fn=views_fn,
                                          z_dim=dim,
                                          proj='MLP',
                                          node_level=False,
                                          **kwargs)

        def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
            # GraphCL removes projection heads after pre-training
            for enc, proj in super(GraphCL, self).train(encoders, data_loader, 
                                                        optimizer, epochs, per_epoch_out):
                yield enc



Note that the :obj:`train` returns a generator the yields trained encoder and projection heads
at each iteration. That is because some contrastive methods also requires the projection heads
in downstream tasks (such as MVGRL).



Evaluation of encapsulated models
---------------------------------

You can always write your own code to do flexible evlauation of the above defined contrastive methods.
However, we provide pre-implemented evluation tools for more convenient evaluation. The tool works with
most datasets from :obj:`pytorch-geometric`. Below is an example of perform semi-supervised evaluation
for GraphCL. More examples can be found in runnable jupyter notebooks in the benchmark.

For the first step, we load the dataset NCI, which is a typical dataset for graph classification.
One can also use different datasets for pretraining and finetuning.

.. code-block ::

    from dig.sslgraph.dataset import get_dataset
    dataset, dataset_pretrain = get_dataset('NCI1', task='semisupervised')
    feat_dim = dataset[0].x.shape[1]
    embed_dim = 128
    

Then we employ ResGCN [4]_ as the graph encoder and run the evaluation.

.. code-block ::

    from dig.sslgraph.utils import Encoder
    from dig.sslgraph.method import GraphCL
    from dig.sslgraph.evaluation import GraphSemisupervised

    encoder = Encoder(feat_dim, embed_dim, n_layers=3, gnn='resgcn')
    graphcl = GraphCL(embed_dim, aug_1='subgraph', aug_2='subgraph')
    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01)
    evaluator.evaluate(learning_model=graphcl, encoder=encoder)



.. [1] Liu, M., Luo, Y., Wang, L., Xie, Y., Yuan, H., Gui, S., Yu, H., Xu, Z., Zhang, J., Liu, Y. and Yan, K., 2021. DIG: A Turnkey Library for Diving into Graph Deep Learning Research. arXiv preprint arXiv:2103.12608.
.. [2] Xie, Y., Xu, Z., Zhang, J., Wang, Z. and Ji, S., 2021. Self-supervised learning of graph neural networks: A unified review. arXiv preprint arXiv:2102.10757.
.. [3] You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z. and Shen, Y., 2020. Graph contrastive learning with augmentations. Advances in Neural Information Processing Systems, 33.
.. [4] Chen, T., Bian, S. and Sun, Y., 2019. Are powerful graph neural nets necessary? a dissection on graph classification. arXiv preprint arXiv:1905.04579.
.. [5] Hassani, K. and Khasahmadi, A.H., 2020, November. Contrastive multi-view representation learning on graphs. In International Conference on Machine Learning (pp. 4116-4126). PMLR.
.. [6] Sun, F.Y., Hoffmann, J., Verma, V. and Tang, J., 2019. Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization. arXiv preprint arXiv:1908.01000.
.. [7] Xu, K., Hu, W., Leskovec, J. and Jegelka, S., 2018. How powerful are graph neural networks?. arXiv preprint arXiv:1810.00826.
.. [8] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y., 2017. Graph attention networks. arXiv preprint arXiv:1710.10903.
.. [9] Kipf, T.N. and Welling, M., 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
