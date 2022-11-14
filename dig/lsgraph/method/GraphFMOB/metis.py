from typing import Tuple

import time
import copy

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data

partition_fn = torch.ops.torch_sparse.partition


def metis(adj_t: SparseTensor, num_parts: int, recursive: bool = False,
          log: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Computes the METIS partition of a given sparse adjacency matrix
    :obj:`adj_t`, returning its "clustered" permutation :obj:`perm` and
    corresponding cluster slices :obj:`ptr`."""

    if log:
        t = time.perf_counter()
        print(f'Computing METIS partitioning with {num_parts} parts...',
              end=' ', flush=True)

    num_nodes = adj_t.size(0)

    if num_parts <= 1:
        perm, ptr = torch.arange(num_nodes), torch.tensor([0, num_nodes])
    else:
        rowptr, col, _ = adj_t.csr()
        cluster = partition_fn(rowptr, col, None, num_parts, recursive)
        cluster, perm = cluster.sort()
        ptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return perm, ptr


def permute(data: Data, perm: Tensor, log: bool = True) -> Data:
    r"""Permutes a :obj:`data` object according to a given permutation
    :obj:`perm`."""

    if log:
        t = time.perf_counter()
        print('Permuting data...', end=' ', flush=True)

    data = copy.copy(data)
    for key, value in data:
        if isinstance(value, Tensor) and value.size(0) == data.num_nodes:
            data[key] = value[perm]
        elif isinstance(value, Tensor) and value.size(0) == data.num_edges:
            raise NotImplementedError
        elif isinstance(value, SparseTensor):
            data[key] = value.permute(perm)

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return data
