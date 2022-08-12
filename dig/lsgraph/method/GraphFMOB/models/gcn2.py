from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import GCN2Conv


import torch
from torch import Tensor
from torch_sparse import SparseTensor
from ..models import ScalableGNN


class GCN2Conv(GCN2Conv):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GCN2Conv, self).__init__(*args, **kwargs)

    def propagation(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None):
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def transformation(self, x: Tensor, x_0: Tensor):
        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0
        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out += torch.addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                               alpha=self.beta)
        return out

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        x = self.propagation(x, edge_index, edge_weight=edge_weight)
        out = self.transformation(x, x_0[:x.size(0)])
        return out

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)


class GCN2(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 theta: float, shared_weights: bool = True,
                 dropout: float = 0.0, drop_input: bool = True,
                 batch_norm: bool = False, residual: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None,
                 device=None,
                 gamma=0.0):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device, gamma)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.gamma = gamma
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            conv = GCN2Conv(hidden_channels, alpha=alpha, theta=theta,
                            layer=i + 1, shared_weights=shared_weights,
                            normalize=False)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        return ModuleList(list(self.convs) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins

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

        x = x_0 = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn, hist in zip(self.convs[:-1], self.bns[:-1], self.histories):
            x_bar_out_batch = P_tilde.spmm(x[:batch_size]).detach()

            h = conv(x, x_0, adj_t)
            if self.batch_norm:
                h = bn(h)
            if self.residual:
                h += x[:h.size(0)]
            x = h.relu_()

            # the batch_norm here will make the gamma=0.0 different from the original one
            if n_id is not None and batch_size is not None:
                with torch.no_grad():
                    if self.batch_norm:
                        bn.eval()
                        his_outbatch_emb = bn(conv.transformation(x_bar_out_batch, x_0[batch_size:])).relu()
                        hist.forward(his_outbatch_emb, n_id[batch_size:])
                        bn.train()
                    else:
                        his_outbatch_emb = conv.transformation(x_bar_out_batch, x_0[batch_size:]).relu()
                        hist.forward(his_outbatch_emb, n_id[batch_size:])

            x = self.push_and_pull(hist, x, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, x_0, adj_t)
        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = h.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[1](x)
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = x_0 = self.lins[0](x).relu_()
            state['x_0'] = x_0

        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[layer](x, state['x_0'], adj_t)
        if self.batch_norm:
            h = self.bns[layer](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        x = h.relu_()

        if layer == self.num_layers - 1:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)

        return x