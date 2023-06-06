.. dig documentation master file, created by
   sphinx-quickstart on Thu Apr 15 09:25:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
:github_url: https://github.com/divelab/DIG

DIG: Dive into Graphs Documentation
====================================

.. image:: ../imgs/DIG-logo.jpg
   :width: 50%


DIG: Dive into Graphs is a turnkey library for graph deep learning research.

Why DIG?
^^^^^^^^

The key difference with current graph deep learning libraries, such as PyTorch Geometric (PyG) and Deep Graph Library (DGL), is that, while PyG and DGL support basic graph deep learning operations, **DIG provides a unified testbed for higher level, research-oriented graph deep learning tasks**, such as graph generation, self-supervised learning, explainability, and 3D graphs.

If you are working or plan to work on research in graph deep learning, DIG enables you to develop your own methods within our extensible framework, and compare with current baseline methods using common datasets and evaluation metrics without extra efforts.



.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Get Started

   intro/introduction
   intro/installation
   


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/graphdf
   tutorials/sslgraph
   tutorials/subgraphx
   tutorials/threedgraph
   tutorials/oodgraph
   tutorials/fairgraph


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Graph Augmentation

   auggraph/dataset
   auggraph/method


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Graph Generation

   ggraph/dataset
   ggraph/method
   ggraph/evaluation
   ggraph/utils


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 3D Graph Generation

   ggraph3d/dataset
   ggraph3d/method
   ggraph3d/evaluation
   ggraph3d/utils


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Self-supervised Learning on Graphs

   sslgraph/dataset
   sslgraph/method
   sslgraph/evaluation
   sslgraph/utils
   
   

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Explainability of GNNs

   xgraph/dataset
   xgraph/method
   xgraph/evaluation
   xgraph/utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deep Learning on 3D Graphs

   3dgraph/dataset
   3dgraph/method
   3dgraph/evaluation
   3dgraph/utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Graph OOD (GOOD) datasets

   oodgraph/good

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Fair graph representation

   fairgraph/dataset
   fairgraph/method

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Contributing

   contribution/instruction
