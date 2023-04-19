# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv
from torch_scatter import scatter
from ..constants import *


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
            self.update_net = torch.nn.GRU(node_feat_dim * 3, node_feat_dim)
        else:
            layer = []
            layer.append(torch.nn.Linear(node_feat_dim * 4, update_net_hiddens[0]))
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


class AugEncoder(torch.nn.Module):
    def __init__(self, in_dim, num_layers, hidden, conv_type=ConvType.GEMB, use_virtual_node=False, node_update_type=NodeUpdateType.RESIDUAL, layer_norm=False):
        super(AugEncoder, self).__init__()
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden))
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == ConvType.GEMB:
                self.convs.append(GEMBConv(hidden, [hidden * 2, hidden * 2], [hidden * 2], node_update_type, layer_norm))
            elif conv_type == ConvType.GIN:
                self.convs.append(
                    GINConv(Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                        train_eps=True))
        
        self.use_virtual_node = use_virtual_node
        if use_virtual_node:
            self.virtual_init = torch.nn.Embedding(1, hidden)
            self.virtual_mlps = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.virtual_mlps.append(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                ))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        if self.use_virtual_node:
            virtual_embed = self.virtual_init(torch.zeros(1).long().to(x.device))
        
        for i, conv in enumerate(self.convs):
            if self.use_virtual_node:
                x = x + virtual_embed
                virtual_embed = torch.sum(x, dim=0, keepdim=True) + virtual_embed
            x = conv(x, edge_index)
            if self.use_virtual_node:
                virtual_embed = self.virtual_mlps[i](virtual_embed)
        
        if self.use_virtual_node:
            return x, virtual_embed
        return x