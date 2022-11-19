# Copyright (c) 2021 Big Data and Multi-modal Computing Group, CRIPAC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adaptive augmentation methods proposed in the paper `Graph Contrastive Learning
# with Adaptive Augmentation <https://arxiv.org/abs/2010.14945>`_. You can refer to `the original implementation 
# <https://github.com/CRIPAC-DIG/GCA>` or `the benchmark code 
# <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_grace.ipynb>`_ for
# an example of usage.

import torch
from torch.utils.data import random_split
from torch_geometric.utils import degree, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx, degree, to_undirected
from torch_scatter import scatter
import networkx as nx

def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    '''Returns the PageRank centrality given the edge_index
    Args:
        edge_index (torch.Tensor): Tensor of shape [2, n_edges]
        damp (float, optional): Damping factor
        k (int, optional): Number of iterations
    '''
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    '''Returns the new node features after probabilistically masking some`
    Args:
        x (torch.Tensor): Tensor of shape [n_nodes, n_features] representing node features.
        w (torch.Tensor): Tensor of shape [n_features] representing weights
        p (float): Probability multiplier
        threshold (float): Upper bound probability of masking a feature
    '''
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    '''Returns the weights of dropping each feature. Meant to be used for sparse one-hot features only.
    Args:
        x (torch.Tensor): Tensor of shape [n_nodes, n_features] representing node features
        node_c (torch.Tensor): Tensor of shape [n_nodes] represnting node centralities
    Returns:
        A Tensor of shape [n_features] representing the weights for dropping each feature
    '''
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    '''Same as `feature_drop_weights` but for dense continuous features
    '''
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float):
    '''Returns the new edge index after probabilistically dropping edges from `edge_index`
    Args:
        edge_index (torch.Tensor): Tensor of shape [2, n_edges]
        edge_weights (torch.Tensor): Tensor of shape [n_edges]
        p (float): Probability multiplier
        threshold (float): Upper bound probability of dropping an edge
    '''
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):
    '''Returns the dropping weight of each edge depending on the degree centrality
    Args:
        edge_index (torch.Tensor): A tensor of shape [2, n_edges]
    :rtype: :class:`Tensor`
    '''
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    '''Returns the dropping weight of each edge depending on the PageRank centrality
    Args:
        edge_index (torch.Tensor): A tensor of shape [2, n_edges]
    :rtype: :class:`Tensor`
    '''
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    '''Returns the dropping weight of each edge depending on the Eigen vector centrality
    Args:
        edge_index (torch.Tensor): A tensor of shape [2, n_edges]
    :rtype: :class:`Tensor`
    '''
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    '''Utility function for generating training, validation and test set samples.
    '''
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask
