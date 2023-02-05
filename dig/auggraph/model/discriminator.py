import torch
import torch.nn as nn
from .genet import GENet
from .gmnet import GMNet

class DiscriminatorModel(torch.nn.Module):
    def __init__(self, in_dim, num_layers, hidden, pool_type='sum', model_type='gmnet', fuse_type='abs_diff', **kwargs):
        super(DiscriminatorModel, self).__init__()
        if model_type == 'gmnet':
            self.dis_encoder = GMNet(in_dim, num_layers, hidden, pool_type=pool_type, **kwargs)
        elif model_type == 'genet':
            self.dis_encoder = GENet(in_dim, num_layers, hidden, pool_type=pool_type, **kwargs)
        
        self.fuse_type = fuse_type
        if fuse_type == 'concat':
            self.pred_head = nn.Sequential(
                nn.Linear(2 * hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, 1),
                nn.Sigmoid(),
            )
        elif fuse_type == 'cos':
            self.pred_head = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, hidden)
            )
            self.cos = torch.nn.CosineSimilarity(dim=1)
        else:
            in_hidden = hidden
            self.pred_head = nn.Sequential(
                nn.Linear(in_hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, 1),
                nn.Sigmoid(),
            )

    def forward(self, data1, data2):
        embed1, embed2 = self.dis_encoder(data1, data2)

        if self.fuse_type == 'add':
            pair_embed = embed1 + embed2
        elif self.fuse_type == 'multiply':
            pair_embed = embed1 * embed2
        elif self.fuse_type == 'concat':
            pair_embed = torch.cat((embed1, embed2), dim=1)
        elif self.fuse_type == 'abs_diff':
            pair_embed = torch.abs(embed1 - embed2)
        elif self.fuse_type == 'cos':
            embed1, embed2 = self.pred_head(embed1), self.pred_head(embed2)
            prob = (1.0 + self.cos(embed1, embed2)) / 2.0
            return prob
        
        prob = self.pred_head(pair_embed)
        return prob