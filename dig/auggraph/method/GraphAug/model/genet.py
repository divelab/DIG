# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GINConv
from dig.auggraph.method.GraphAug.constants import *


class GEMBConv(torch.nn.Module):
    def __init__(self, node_feat_dim, message_net_hiddens, update_net_hiddens, node_update_type=NodeUpdateType.RESIDUAL, layer_norm=False):
        super(GEMBConv, self).__init__()
        assert node_update_type in [NodeUpdateType.MLP, NodeUpdateType.RESIDUAL, NodeUpdateType.GRU]
        self.node_update_type = node_update_type
        self._get_message_net(node_feat_dim, message_net_hiddens)
        self._get_update_net(node_feat_dim, update_net_hiddens, node_update_type)
        if layer_norm:
            self.message_norm = torch.nn.LayerNorm()
            self.update_norm = torch.nn.LayerNorm()

    def _get_message_net(self, node_feat_dim, message_net_hiddens):
        layer = []
        layer.append(torch.nn.Linear(node_feat_dim * 2, message_net_hiddens[0]))
        for i in range(1, len(message_net_hiddens)):
            layer.append(torch.nn.ReLU())
            layer.append(torch.nn.Linear(message_net_hiddens[i - 1], message_net_hiddens[i]))
        self.message_net = torch.nn.Sequential(*layer)

    def _get_update_net(self, node_feat_dim, update_net_hiddens, node_update_type):
        if node_update_type == NodeUpdateType.GRU:
            self.update_net = torch.nn.GRU(node_feat_dim * 2, node_feat_dim)
        else:
            layer = []
            layer.append(torch.nn.Linear(node_feat_dim * 3, update_net_hiddens[0]))
            for i in range(1, len(update_net_hiddens)):
                layer.append(torch.nn.ReLU())
                layer.append(torch.nn.Linear(update_net_hiddens[i - 1], update_net_hiddens[i]))
            layer.append(torch.nn.ReLU())
            layer.append(torch.nn.Linear(update_net_hiddens[-1], node_feat_dim))
            self.update_net = torch.nn.Sequential(*layer)

    def message_aggr(self, x, edge_index):
        target_idx, source_idx = edge_index
        message_inputs = torch.cat((x.index_select(0, source_idx), x.index_select(0, target_idx)), dim=-1)
        messages = self.message_net(message_inputs)
        aggregation = scatter(messages, target_idx, dim=0, dim_size=x.shape[0], reduce=ReduceType.ADD.value)

        if hasattr(self, 'message_norm'):
            aggregation = self.message_norm(aggregation)
        return aggregation

    def node_update(self, x, messages):
        if self.node_update_type == NodeUpdateType.GRU:
            _, new_x = self.update_net(messages.unsqueeze(0), x.unsqueeze(0))
            new_x = torch.squeeze(new_x)
        else:
            update_inputs = torch.cat((messages, x), dim=-1)
            new_x = self.update_net(update_inputs)

        if hasattr(self, 'update_norm'):
            new_x = self.update_norm(new_x)

        if self.node_update_type == NodeUpdateType.RESIDUAL:
            return x + new_x
        return new_x

    def forward(self, x, edge_index):
        messages = self.message_aggr(x, edge_index)
        new_x = self.node_update(x, messages)
        return new_x


class Readout(nn.Module):
    def __init__(self, node_feat_dim, node_hiddens, graph_hiddens, use_gate=True, pool_type=PoolType.SUM):
        super(Readout, self).__init__()
        self.graph_feat_dim = node_hiddens[-1]
        self.use_gate = use_gate
        self._get_node_net(node_feat_dim, node_hiddens, use_gate)
        self._get_graph_net(self.graph_feat_dim, graph_hiddens)

        if pool_type == PoolType.MEAN:
            self.pool = global_mean_pool
        elif pool_type == PoolType.SUM:
            self.pool = global_add_pool
        elif pool_type == PoolType.MAX:
            self.pool = global_max_pool

    def _get_node_net(self, node_feat_dim, node_hiddens, use_gate):
        if use_gate:
            node_hiddens[-1] *= 2

        layer = []
        layer.append(nn.Linear(node_feat_dim, node_hiddens[0]))
        for i in range(1, len(node_hiddens)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hiddens[i - 1], node_hiddens[i]))

        self.node_net = nn.Sequential(*layer)

    def _get_graph_net(self, graph_feat_dim, graph_hiddens):
        layer = []
        layer.append(nn.Linear(graph_feat_dim, graph_hiddens[0]))
        for i in range(1, len(graph_hiddens)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(graph_hiddens[i - 1], graph_hiddens[i]))

        self.graph_net = nn.Sequential(*layer)

    def forward(self, x, batch):
        x = self.node_net(x)
        if self.use_gate:
            gates = torch.sigmoid(x[:, :self.graph_feat_dim])
            x = x[:, self.graph_feat_dim:] * gates
        graph_feat = self.pool(x, batch)
        graph_feat = self.graph_net(graph_feat)

        return graph_feat


class GENet(nn.Module):
    def __init__(self, in_dim, num_layers, hidden, conv_type=ConvType.GEMB, pool_type=PoolType.SUM, use_gate=True,
                 node_update_type=NodeUpdateType.RESIDUAL, layer_norm=False):
        super(GENet, self).__init__()

        self.embedding = nn.Sequential(nn.Linear(in_dim, hidden))

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == ConvType.GEMB:
                self.convs.append(GEMBConv(hidden, [hidden * 2, hidden * 2], [hidden * 2], node_update_type, layer_norm))
            elif conv_type == ConvType.GIN:
                self.convs.append(
                    GINConv(nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BN(hidden),
                    ), train_eps=True))

        self.readout = Readout(hidden, [hidden], [hidden], use_gate, pool_type)

    def embed(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        x1, x2 = self.embedding(x1), self.embedding(x2)
        for conv in self.convs:
            x1, x2 = conv(x1, edge_index1), conv(x2, edge_index2)

        embed1, embed2 = self.readout(x1, batch1), self.readout(x2, batch2)
        return embed1, embed2

    def forward(self, data1, data2):
        embed1, embed2 = self.embed(data1, data2)
        return embed1, embed2
