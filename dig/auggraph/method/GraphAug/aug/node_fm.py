# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.data import Data
from ..constants import *

class NodeFM(torch.nn.Module):
    def __init__(self, training=False, uniform=False, hid_dim=128, node_feat_dim=128, magnitude=None, temperature=1.0, mask_type=MaskType.ZERO):
        super(NodeFM, self).__init__()
        self.mask_type = mask_type
        self.uniform = uniform
        self.magnitude = magnitude
        if not uniform:
            self.node_feat_mask_mlp = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, 2 * hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * hid_dim, node_feat_dim)
            )
            self.training = training
            self.temperature = temperature
        else:
            assert magnitude is not None
    
    def reset_parameters(self):
        self.node_feat_mask_mlp[0].reset_parameters()
        self.node_feat_mask_mlp[2].reset_parameters()
    
    def uniform_node_feat_mask(self, data):
        x = data.x.detach().clone()
        drop_mask = torch.empty(x.shape, dtype=torch.float32).uniform_(0, 1) < self.magnitude
        
        if self.mask_type == MaskType.ZERO:
            x[drop_mask] = 0
        elif self.mask_type == MaskType.GAUSSIAN:
            x[drop_mask] = x[drop_mask].normal_(0.0, 0.5)
        
        new_data = Data(x=x, edge_index=data.edge_index, y=data.y)
        return new_data, None
    
    def node_feat_mask(self, data, h):
        # print(data.x[:,1])
        logits = self.node_feat_mask_mlp(h)
        sample_logits = logits.detach() * self.temperature
        mask = Bernoulli(logits=sample_logits).sample()

        if self.magnitude is not None:
            keep_mask = torch.empty(data.x.shape, dtype=torch.float32).uniform_(0, 1) > self.magnitude
            mask[keep_mask] = 0.0
        
        x = data.x.detach().clone()
        if self.mask_type == MaskType.ZERO:
            x = x * (1.0 - mask)
        elif self.mask_type == MaskType.GAUSSIAN:
            mask = mask.bool()
            x[mask] = x[mask].normal_(0.0, 0.5)
        
        new_data = Data(x=x, edge_index=data.edge_index, y=data.y)

        if self.training:
            log_likelihood = - F.binary_cross_entropy_with_logits(logits * self.temperature, mask, reduction='sum')
            return new_data, log_likelihood

        return new_data, None

    def forward(self, data, h):
        if self.uniform:
            return self.uniform_node_feat_mask(data)
        return self.node_feat_mask(data, h)