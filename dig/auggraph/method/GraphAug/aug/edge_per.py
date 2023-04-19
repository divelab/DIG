# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import torch
import random
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.data import Data
from torch_geometric.utils.dropout import filter_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from .aug_utils import coalesce


def negative_sampling(edge_index, num_nodes, num_neg_edge, undirected=False):
    row, col = edge_index
    adj_dense = torch.zeros([num_nodes, num_nodes], device=row.device)
    adj_dense[row, col] = 1

    neg_adj_dense = 1.0 - adj_dense
    neg_row, neg_col = torch.nonzero(neg_adj_dense).t()
    if undirected:
        neg_row, neg_col, _ = filter_adj(neg_row, neg_col, None, neg_row < neg_col)
    else:
        neg_row, neg_col, _ = filter_adj(neg_row, neg_col, None, neg_row != neg_col)
    
    num_neg_edge = min(num_neg_edge, len(neg_row))
    neg_idx = random.sample(range(len(neg_row)), num_neg_edge)
    return neg_row[neg_idx], neg_col[neg_idx]


class EdgePer(torch.nn.Module):
    def __init__(self, training=False, only_drop=False, uniform=False, hid_dim=128, magnitude=None, temperature=1.0, undirected=True):
        super(EdgePer, self).__init__()
        self.uniform = uniform
        self.magnitude = magnitude
        self.undirected = undirected
        if not uniform:
            self.edge_per_mlp = torch.nn.Sequential(
                torch.nn.Linear(hid_dim + 1, 2 * hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * hid_dim, 1)
            )
            self.only_drop = only_drop
            self.training = training
            self.temperature = temperature
        else:
            assert magnitude is not None
            if only_drop:
                self.uniform_per_func = self.uniform_edge_drop
            else:
                self.uniform_per_func = self.uniform_edge_per
        
    def reset_parameters(self):
        self.edge_per_mlp[0].reset_parameters()
        self.edge_per_mlp[2].reset_parameters()
    
    def uniform_edge_per(self, data):
        edge_index = data.edge_index
        row, col = edge_index
        # if self.undirected:
        #     row, col, _ = filter_adj(row, col, None, row < col)
        
        num_nodes = maybe_num_nodes(edge_index, num_nodes=data.x.shape[0])
        adj_dense = torch.zeros([num_nodes, num_nodes])
        adj_dense[row, col] = 1
        per_mask = adj_dense.new_full(adj_dense.shape, self.magnitude, dtype=torch.float)
        per_mask = torch.bernoulli(per_mask)
        # per_mask[adj_dense == 0] = 0
        
        # print(torch.nonzero(per_mask))
        new_adj_dense = per_mask - adj_dense

        new_row, new_col = torch.nonzero(new_adj_dense).t()
        if self.undirected:
            new_row, new_col, _ = filter_adj(new_row, new_col, None, new_row < new_col)
            new_edge_index = torch.stack([torch.cat([new_row, new_col], dim=0), torch.cat([new_col, new_row], dim=0)], dim=0)
        else:
            new_edge_index = torch.cat([new_row, new_col], dim=0)
        new_edge_index = coalesce(new_edge_index, None, num_nodes=num_nodes)
        
        new_data = Data(x=data.x, edge_index=new_edge_index.to(edge_index.device), y=data.y)
        return new_data, None

    def uniform_edge_drop(self, data):
        edge_index = data.edge_index.detach().clone()
        row, col = edge_index
        if self.undirected:
            row, col, _ = filter_adj(row, col, None, row < col)
        
        idxs = random.sample(range(len(row)), int((1.0 - self.magnitude) * len(row)))
        new_row, new_col = row[idxs], col[idxs]

        if self.undirected:
            new_edge_index = torch.stack([torch.cat([new_row, new_col], dim=0), torch.cat([new_col, new_row], dim=0)], dim=0)
        else:
            new_edge_index = torch.cat([new_row, new_col], dim=0)
        new_edge_index = coalesce(new_edge_index, None, num_nodes=data.x.shape[0])

        new_data = Data(x=data.x, edge_index=new_edge_index.to(edge_index.device), y=data.y)
        return new_data, None
    
    def edge_per(self, data, h):
        pos_edge_index = data.edge_index.detach().clone()
        pos_row, pos_col = pos_edge_index
        if self.undirected:
            pos_row, pos_col, _ = filter_adj(pos_row, pos_col, None, pos_row < pos_col)
        
        drop_logits = self.edge_per_mlp(torch.cat((h[pos_row] + h[pos_col], torch.ones([len(pos_row), 1], device=h.device)), dim=-1))
        drop_sample_logits = drop_logits.detach().view(-1) * self.temperature
        drop_mask = Bernoulli(logits=drop_sample_logits).sample()
        
        if self.magnitude is not None:
            keep_mask = torch.empty([len(drop_mask)], dtype=torch.float32).uniform_(0, 1) > self.magnitude
            drop_mask[keep_mask] = 0.0

        pos_row, pos_col = pos_row[(1.0 - drop_mask).bool()], pos_col[(1.0 - drop_mask).bool()]

        if self.only_drop:
            new_row, new_col = pos_row, pos_col
        else:
            neg_row, neg_col = negative_sampling(data.edge_index, num_nodes=data.x.shape[0], num_neg_edge=pos_edge_index.shape[1], undirected=self.undirected)
            add_logits = self.edge_per_mlp(torch.cat((h[neg_row] + h[neg_col], torch.zeros([len(neg_row), 1], device=h.device)), dim=-1))
            add_sample_logits = add_logits.detach().view(-1) * self.temperature
            add_mask = Bernoulli(logits=add_sample_logits).sample()

            if self.magnitude is not None:
                keep_mask = torch.empty([len(add_mask)], dtype=torch.float32).uniform_(0, 1) > self.magnitude
                add_mask[keep_mask] = 0.0
                    
            neg_row, neg_col = neg_row[add_mask.bool()], neg_col[add_mask.bool()]
            new_row, new_col = torch.cat((pos_row, neg_row), dim=-1), torch.cat((pos_col, neg_col), dim=-1)
        
        if self.undirected:
            new_edge_index = torch.stack([torch.cat([new_row, new_col], dim=0), torch.cat([new_col, new_row], dim=0)], dim=0)
        else:
            new_edge_index = torch.cat([new_row, new_col], dim=0)
        new_edge_index = coalesce(new_edge_index, None, num_nodes=data.x.shape[0])

        new_data = Data(x=data.x, edge_index=new_edge_index, y=data.y)
        
        if self.training:
            if self.only_drop:
                log_likelihood = - F.binary_cross_entropy_with_logits(drop_logits.view(-1) * self.temperature, drop_mask, reduction='sum')
            else:
                log_likelihood = - F.binary_cross_entropy_with_logits(drop_logits.view(-1) * self.temperature, drop_mask, reduction='sum') \
                    - F.binary_cross_entropy_with_logits(add_logits.view(-1), add_mask, reduction='sum')
            return new_data, log_likelihood

        return new_data, None

    def forward(self, data, h):
        if self.uniform:
            return self.uniform_per_func(data)
        return self.edge_per(data, h)