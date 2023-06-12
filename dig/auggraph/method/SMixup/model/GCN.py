import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from typing import List, Optional, Union
from torch import Tensor
from torch_scatter import scatter

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN


class GCN(torch.nn.Module):
    def __init__(self, in_dim, num_classes, num_layers, hidden, dropout=0.5, pool_type='mean', use_jk=False, jk_mode='cat'):
        super(GCN, self).__init__()
        num_features = in_dim
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        
        self.use_jk = use_jk
        if use_jk:
            self.jk = JumpingKnowledge(jk_mode)

        lin_in_dim = num_layers * hidden if use_jk and jk_mode == 'cat' else hidden
        self.lin1 = Linear(lin_in_dim, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.dropout = dropout

        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'sum':
            self.pool = global_add_pool
        elif pool_type == 'max':
            self.pool = global_max_pool

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.use_jk:
            self.jk.reset_parameters()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if 'edge_weights' in data:
            edge_weight = data.edge_weights.float()
        else:
            edge_weight = None
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight = edge_weight))
            xs += [x]
        
        if self.use_jk:
            x = self.jk(xs)
        h = self.pool(x, batch)

        x = F.relu(self.lin1(h))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return h, x
        # return F.log_softmax(x, dim=-1)
        
    def M_mixup_forward(self, data1, data2, lambd):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        
        xs = [x1]
        x1 = F.relu(self.conv1(x1, edge_index1))
        for conv in self.convs:
            x1 = F.relu(conv(x1, edge_index1))
            xs += [x1]
        if self.use_jk:
            x1 = self.jk(xs)
        embed1 = self.pool(x1, batch1)
        
        xs = [x2]
        x2 = F.relu(self.conv1(x2, edge_index2))
        for conv in self.convs:
            x2 = F.relu(conv(x2, edge_index2))
            xs += [x2]
        if self.use_jk:
            x2 = self.jk(xs)
        embed2 = self.pool(x2, batch2)
        
        mixup_embed = lambd * embed1 + (1.0 - lambd) * embed2
        embed = F.relu(self.lin1(mixup_embed))
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        embed = self.lin2(embed)
        # return F.log_softmax(embed, dim=-1)
        
        return mixup_embed, embed

    def __repr__(self):
        return self.__class__.__name__
  