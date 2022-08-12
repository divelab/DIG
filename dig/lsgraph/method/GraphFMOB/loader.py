from typing import NamedTuple, List, Tuple

import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from dig_ext.relabel import relabel_one_hop

relabel_fn = relabel_one_hop


class SubData(NamedTuple):
    data: Data
    batch_size: int
    n_id: Tensor  # The indices of mini-batched nodes
    offset: Tensor  # The offset of contiguous mini-batched nodes
    count: Tensor  # The number of contiguous mini-batched nodes

    def to(self, *args, **kwargs):
        return SubData(self.data.to(*args, **kwargs), self.batch_size,
                       self.n_id, self.offset, self.count)


class SubgraphLoader(DataLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors)."""
    def __init__(self, data: Data, ptr: Tensor, batch_size: int = 1,
                 bipartite: bool = True, log: bool = True, **kwargs):

        self.data = data
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log

        n_id = torch.arange(data.num_nodes)
        batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
        batches = [(i, batches[i]) for i in range(len(batches))]

        if batch_size > 1:
            super().__init__(batches, batch_size=batch_size,
                             collate_fn=self.compute_subgraph, **kwargs)

        else:  # If `batch_size=1`, we pre-process the subgraph generation:
            if log:
                t = time.perf_counter()
                print('Pre-processing subgraphs...', end=' ', flush=True)

            data_list = list(
                DataLoader(batches, collate_fn=self.compute_subgraph,
                           batch_size=batch_size, **kwargs))

            if log:
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

            super().__init__(data_list, batch_size=batch_size,
                             collate_fn=lambda x: x[0], **kwargs)

    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        batch_ids, n_ids = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)

        # We collect the in-mini-batch size (`batch_size`), the offset of each
        # partition in the mini-batch (`offset`), and the number of nodes in
        # each partition (`count`)
        batch_size = n_id.numel()
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset)

        rowptr, col, value = self.data.adj_t.csr()
        rowptr, col, value, n_id = relabel_fn(rowptr, col, value, n_id,
                                              self.bipartite)

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)

        data = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)

        return SubData(data, batch_size, n_id, offset, count)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class EvalSubgraphLoader(SubgraphLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors).
    In contrast to :class:`SubgraphLoader`, this loader does not generate
    subgraphs from randomly sampled mini-batches, and should therefore only be
    used for evaluation.
    """
    def __init__(self, data: Data, ptr: Tensor, batch_size: int = 1,
                 bipartite: bool = True, log: bool = True, **kwargs):

        ptr = ptr[::batch_size]
        if int(ptr[-1]) != data.num_nodes:
            ptr = torch.cat([ptr, torch.tensor([data.num_nodes])], dim=0)

        super().__init__(data=data, ptr=ptr, batch_size=1, bipartite=bipartite,
                         log=log, shuffle=False, num_workers=0, **kwargs)
