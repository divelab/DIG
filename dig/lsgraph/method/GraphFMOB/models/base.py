"""
Refer from https://github.com/rusty1s/pyg_autoscale/blob/master/torch_geometric_autoscale/models/base.py
"""

from typing import Optional, Callable, Dict, Any

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from dig.lsgraph.method.FM import FeatureMomentum
from ..pool import AsyncIOPool
from ..loader import SubgraphLoader, EvalSubgraphLoader


class ScalableGNN(torch.nn.Module):
    r"""An abstract class for implementing scalable GNNs via historical
    embeddings.
    This class will take care of initializing :obj:`num_layers - 1` historical
    embeddings, and provides a convenient interface to push recent node
    embeddings to the history, and to pull previous embeddings from the
    history.
    In case historical embeddings are stored on the CPU, they will reside
    inside pinned memory, which allows for asynchronous memory transfers of
    historical embeddings.
    For this, this class maintains a :class:`AsyncIOPool` object that
    implements the underlying mechanisms of asynchronous memory transfers as
    described in our paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        hidden_channels (int): The number of hidden channels of the model.
            As a current restriction, all intermediate node embeddings need to
            utilize the same number of features.
        num_layers (int): The number of layers of the model.
        pool_size (int, optional): The number of pinned CPU buffers for pulling
            histories and transfering them to GPU.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
        buffer_size (int, optional): The size of pinned CPU buffers, i.e. the
            maximum number of out-of-mini-batch nodes pulled at once.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
    """
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None,
                 device=None,
                 gamma=0.0):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers if pool_size is None else pool_size
        self.buffer_size = buffer_size
        self.gamma = gamma

        self.histories = torch.nn.ModuleList([
            FeatureMomentum(num_nodes, hidden_channels, device, gamma)
            for _ in range(num_layers - 1)
        ])

        self.pool: Optional[AsyncIOPool] = None
        self._async = False
        self.__out: Optional[Tensor] = None

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.histories[0]._device

    def _apply(self, fn: Callable) -> None:
        super()._apply(fn)
        # We only initialize the AsyncIOPool in case histories are on CPU:
        if (str(self.emb_device) == 'cpu' and str(self.device)[:4] == 'cuda'
                and self.pool_size is not None
                and self.buffer_size is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories[0].embedding_dim)
            self.pool.to(self.device)
        return self

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()

    def __call__(
        self,
        x: Optional[Tensor] = None,
        adj_t: Optional[SparseTensor] = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        loader: EvalSubgraphLoader = None,
        **kwargs,
    ) -> Tensor:
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj:`3` and :obj:`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 2, 5]
            count = [2, 3]

        Args:
            x (Tensor, optional): Node feature matrix. (default: :obj:`None`)
            adj_t (SparseTensor, optional) The sparse adjacency matrix.
                (default: :obj:`None`)
            batch_size (int, optional): The in-mini-batch size of nodes.
                (default: :obj:`None`)
            n_id (Tensor, optional): The global indices of mini-batched and
                out-of-mini-batched nodes. (default: :obj:`None`)
            offset (Tensor, optional): The offset of mini-batched nodes inside
                a utilize a contiguous memory layout. (default: :obj:`None`)
            count (Tensor, optional): The number of mini-batched nodes inside a
                contiguous memory layout. (default: :obj:`None`)
            loader (EvalSubgraphLoader, optional): A subgraph loader used for
                evaluating the given GNN in a layer-wise fashsion.
        """

        if loader is not None:
            return self.mini_inference(loader)

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        if self._async:
            for hist in self.histories:
                self.pool.async_pull(hist.emb, None, None, n_id[batch_size:])

        out = self.forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)

        if self._async:
            for hist in self.histories:
                self.pool.synchronize_push()

        self._async = False

        return out

    def push_and_pull(self, history, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""

        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            history.push(x)
            return x

        assert n_id is not None

        if batch_size is None:
            history.push(x, n_id)
            return x

        if not self._async:
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            h = history.pull(n_id[batch_size:])
            return torch.cat([x[:batch_size], h], dim=0)

        else:
            out = self.pool.synchronize_pull()[:n_id.numel() - batch_size]
            self.pool.async_push(x[:batch_size], offset, count, history.emb)
            out = torch.cat([x[:batch_size], out], dim=0)
            self.pool.free_pull()
            return out

    @property
    def _out(self):
        if self.__out is None:
            self.__out = torch.empty(self.num_nodes, self.out_channels,
                                     pin_memory=True)
        return self.__out

    @torch.no_grad()
    def mini_inference(self, loader: SubgraphLoader) -> Tensor:
        r"""An implementation of layer-wise evaluation of GNNs.
        For each individual layer and mini-batch, :meth:`forward_layer` takes
        care of computing the next state of node embeddings.
        Additional state (such as residual connections) can be stored in
        a `state` directory."""

        # We iterate over the loader in a layer-wise fashsion.
        # In order to re-use some intermediate representations, we maintain a
        # `state` dictionary for each individual mini-batch.

        loader = [sub_data + ({}, ) for sub_data in loader]

        # We push the outputs of the first layer to the history:
        for data, batch_size, n_id, offset, count, state in loader:
            x = data.x.to(self.device)
            adj_t = data.adj_t.to(self.device)
            out = self.forward_layer(0, x, adj_t, state)[:batch_size]
            self.pool.async_push(out, offset, count, self.histories[0].emb)
        self.pool.synchronize_push()

        for i in range(1, len(self.histories)):
            # Pull the complete layer-wise history:
            for _, batch_size, n_id, offset, count, _ in loader:
                self.pool.async_pull(self.histories[i - 1].emb, offset, count,
                                     n_id[batch_size:])

            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for batch, batch_size, n_id, offset, count, state in loader:
                adj_t = batch.adj_t.to(self.device)
                x = self.pool.synchronize_pull()[:n_id.numel()]

                out = self.forward_layer(i, x, adj_t, state)[:batch_size]
                self.pool.async_push(out, offset, count, self.histories[i].emb)
                self.pool.free_pull()
            self.pool.synchronize_push()

        # We pull the histories from the last layer:
        for _, batch_size, n_id, offset, count, _ in loader:
            self.pool.async_pull(self.histories[-1].emb, offset, count,
                                 n_id[batch_size:])

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        for batch, batch_size, n_id, offset, count, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            out = self.forward_layer(self.num_layers - 1, x, adj_t,
                                     state)[:batch_size]
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool.synchronize_push()

        return self._out

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor,
                      state: Dict[str, Any]) -> Tensor:
        raise NotImplementedError
