from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv

from ..models import ScalableGNN
from torch_geometric.typing import Adj, OptTensor


class GCNConv(GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConv, self).__init__(* args, **kwargs)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        # out = self.lin(out)
        out = out @ self.weight

        if self.bias is not None:
            out += self.bias

        return out

    def propagation(self, x, edge_index, edge_weight=None):
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return x

    def transformation(self, x):
        # x = self.lin(x)
        x = x @ self.weight

        if self.bias is not None:
            x += self.bias
        return x


class GCN(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None,
                 gamma=0.0):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device, gamma)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual
        self.linear = linear
        self.gamma = gamma

        self.lins = ModuleList()
        if linear:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0 and not linear:
                in_dim = in_channels
            if i == num_layers - 1 and not linear:
                out_dim = out_channels
            conv = GCNConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        if self.linear:
            return ModuleList(list(self.convs) + list(self.bns))
        else:
            return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins if self.linear else self.convs[-1:]

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if len(args) > 0:
            batch_size = args[0]
            n_id = args[1]
            P_bar = adj_t[:, batch_size:]
            P_tilde = P_bar.t()
        else:
            print("no batch_size in the forward of conv")

        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        for layer_idx, (conv, bn, hist) in enumerate(zip(self.convs[:-1], self.bns, self.histories)):
            x_bar_out_batch = P_tilde.spmm(x[:batch_size]).detach()

            h = conv(x, adj_t)
            h = bn(h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = h.relu_()

            if n_id is not None and batch_size is not None:
                with torch.no_grad():
                    if self.batch_norm:
                        bn.eval()
                        x_bar_out_batch = bn(conv.transformation(x_bar_out_batch)).relu()
                        hist.forward(x_bar_out_batch, n_id[batch_size:])
                        bn.train()
                    else:
                        x_bar_out_batch = conv.transformation(x_bar_out_batch).relu()
                        hist.forward(x_bar_out_batch, n_id[batch_size:])

            x = self.push_and_pull(hist, x, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, adj_t)

        if not self.linear:
            return h

        if self.batch_norm:
            h = self.bns[-1](h)

        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[1](h)

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.linear:
                x = self.lins[0](x).relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](x, adj_t)

        if layer < self.num_layers - 1 or self.linear:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()

        if self.linear:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        return h
