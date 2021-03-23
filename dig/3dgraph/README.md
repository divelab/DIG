# Deep Learning on 3D Graphs

## Overview

The 3dgraph package is a collection of benchmark datasets, data interfaces, evaluation metrics, and state-of-the-art algorithms for 3D graphs. We aims to provide methods under a unified framework, datasets and evaluation metrics for academic researchers interested in 3D graphs.  

## Implemented Algorithms

The `3dgraph` package implements three state-of-the-art algorithms under the [3DGN framework](https://github.com/divelab/DIG/tree/main/dig/3dgraph/3dgn) and offers detailed code running instructions. The information about the three algorithms is summarized in the following table.

| Method | Links | Brief description |
| ------ | ----- | ------------------ |
| SchNet| [Paper](https://arxiv.org/abs/1706.08566) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/3dgraph/schnet) | SchNet uses continuous-ﬁlter convolutional layers to model local correlations of data. It essentially incorporates relative distances based on positions.|
| DimeNet++ | [Paper](https://arxiv.org/abs/2011.14115) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/3dgraph/dimenetpp) | DimeNet++ and its older version [DimeNet](https://arxiv.org/abs/2003.03123) explicitly includes distances between nodes and angles between directed edges in the proposed directional message passing process.|
| SphereNet | [Paper](https://arxiv.org/abs/2102.05013) <br> [Code](https://github.com/divelab/DIG/tree/main/dig/3dgraph/spherenet) | SphereNet is based upon the spherical message passing scheme and incorporates both the distance, angle and torsion information during the information aggregation stage.|

## Package Usage

Here we provide some examples of using the data interfaces and evaluation metrics. The detail of running the code can be found in README.md for each methods.

(1) Data interfaces

We provide unified data interfaces for reading benchmark datasets. For 3D data like molecules, the loaded data contains atom type and positions.

```python
from utils import load_qm9
from torch_geometric.data import DataLoader

train_dataset, val_dataset, test_dataset = load_qm9(dataset='qm9', target='U0', train_size=110000, val_size=10000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

(2) Evaluation metrics

```python
from utils import compute_mae, compute_rmse

regression_metric = compute_mae(targets, preds, num_tasks)
```

## Contact
*If you have any questions, please submit a new issue or contact us at Limei Wang [limei@tamu.edu] and Shuiwang Ji [sji@tamu.edu] .*

