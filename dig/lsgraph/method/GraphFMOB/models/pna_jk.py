from typing import Optional, List

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import (ModuleList, Linear, BatchNorm1d, Sequential, ReLU,
                      Identity)
from torch_sparse import SparseTensor

from ..models import ScalableGNN
from ..models.pna import PNAConv


class PNA_JK(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, aggregators: List[int],
                 scalers: List[int], deg: Tensor, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None,
                 gamma=0.0,
                 ):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual
        self.gamma = gamma

        self.lins = ModuleList()
        self.lins.append(
            Sequential(
                Linear(in_channels, hidden_channels),
                BatchNorm1d(hidden_channels) if batch_norm else Identity(),
                ReLU(inplace=True),
            ))
        self.lins.append(
            Linear((num_layers + 1) * hidden_channels, out_channels))

        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = PNAConv(hidden_channels, hidden_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg)
            self.convs.append(conv)

        self.bns = ModuleList()
        for _ in range(num_layers):
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

        x = self.lins[0](x)
        xs = [x[:adj_t.size(0)]]

        for conv, bn, hist in zip(self.convs[:-1], self.bns[:-1], self.histories):
            h = conv(x, adj_t)
            if self.batch_norm:
                h = bn(h)
            if self.residual:
                h += x[:h.size(0)]
            h = h.relu_()

            if n_id is not None and batch_size is not None:
                with torch.no_grad():
                    if self.batch_norm:
                        bn.eval()
                        to_outbatch_emb = conv.propagation(x[:batch_size].detach(), P_tilde)
                        outbatch_emb = conv.transformation(x[batch_size:].detach(), emb=to_outbatch_emb)
                        outbatch_emb = bn(outbatch_emb).relu()
                        hist.forward(outbatch_emb, n_id[batch_size:])
                        bn.train()
                    else:
                        to_outbatch_emb = conv.propagation(x[:batch_size].detach(), P_tilde)
                        outbatch_emb = conv.transformation(x[batch_size:].detach(), emb=to_outbatch_emb).relu()
                        hist.forward(outbatch_emb, n_id[batch_size:])

            xs += [h]
            x = self.push_and_pull(hist, h, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, adj_t)
        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = h.relu_()

        xs += [x]

        x = torch.cat(xs, dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[1](x)

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        # We keep the skip connections in GPU memory for now. If one encounters
        # GPU memory problems, it is advised to push `state['xs']` to the CPU.
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.lins[0](x)
            state['xs'] = [x[:adj_t.size(0)]]

        h = self.convs[layer](x, adj_t)
        if self.batch_norm:
            h = self.bns[layer](h)
        if self.residual:
            h += x[:h.size(0)]
        h = h.relu_()
        state['xs'] += [h]
        h = F.dropout(h, p=self.dropout, training=self.training)

        if layer == self.num_layers - 1:
            h = torch.cat(state['xs'], dim=-1)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        return h
