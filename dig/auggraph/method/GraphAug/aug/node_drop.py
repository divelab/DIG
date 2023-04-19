# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


class NodeDrop(torch.nn.Module):
    def __init__(self, training=False, uniform=False, hid_dim=128, magnitude=None, temperature=1.0):
        super(NodeDrop, self).__init__()
        self.uniform = uniform
        self.magnitude = magnitude
        if not uniform:
            self.node_drop_mlp = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, 2 * hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * hid_dim, 1)
            )
            self.training = training
            self.temperature = temperature
        else:
            assert magnitude is not None
    
    def reset_parameters(self):
        self.node_drop_mlp[0].reset_parameters()
        self.node_drop_mlp[2].reset_parameters()
    
    def uniform_node_drop(self, data):
        num_nodes = maybe_num_nodes(data.edge_index, num_nodes=data.x.shape[0])
        keep_probs = torch.tensor([1.0 - self.magnitude for _ in range(num_nodes)])
        dist = Bernoulli(keep_probs)
        subset = dist.sample().to(torch.bool)

        while subset.sum() == 0:
            dist = Bernoulli(keep_probs)
            subset = dist.sample().to(torch.bool)
        edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x
        if x is not None:
            x = x[subset]
        new_data = Data(x=x, edge_index=edge_index, y=data.y)
        return new_data, None
    
    def node_drop(self, data, h):
        drop_logits = self.node_drop_mlp(h)
        sample_logits = drop_logits.detach().view(-1) * self.temperature
        # sample_logits[sample_logits > 0] = sample_logits[sample_logits > 0] * self.temperature
        drop_mask = Bernoulli(logits=sample_logits).sample()
        
        if self.magnitude is not None:
            keep_mask = torch.empty([len(drop_mask)], dtype=torch.float32).uniform_(0, 1) > self.magnitude
            drop_mask[keep_mask] = 0.0

        count = 0
        while drop_mask.sum().item() >= len(drop_mask) - 1:
            if count == 10:
                break
            drop_mask = Bernoulli(logits=sample_logits).sample()
            count += 1
        if count == 10:
            drop_mask = torch.zeros([len(drop_logits)]).to(drop_logits.device)

        edge_index, _ = subgraph((1.0 - drop_mask).bool(), data.edge_index, relabel_nodes=True)
        x = data.x
        if x is not None:
            x = x[(1.0 - drop_mask).bool()]
        new_data = Data(x=x, edge_index=edge_index, y=data.y)

        if self.training:
            log_likelihood = - F.binary_cross_entropy_with_logits(drop_logits.view(-1) * self.temperature, drop_mask, reduction='sum')
            return new_data, log_likelihood

        return new_data, None

    def forward(self, data, h):
        if self.uniform:
            return self.uniform_node_drop(data)    
        return self.node_drop(data, h)