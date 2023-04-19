# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import torch
import torch.nn as nn
from .genet import GENet
from .gmnet import GMNet
from dig.auggraph.method.GraphAug.constants import *


class RewardGenModel(torch.nn.Module):
    def __init__(self, in_dim, num_layers, hidden, pool_type=PoolType.SUM, model_type=RewardGenModelType.GMNET, fuse_type=FuseType.ABS_DIFF, **kwargs):
        super(RewardGenModel, self).__init__()
        if model_type == RewardGenModelType.GMNET:
            self.reward_gen_encoder = GMNet(in_dim, num_layers, hidden, pool_type=pool_type, **kwargs)
        elif model_type == RewardGenModelType.GENET:
            self.reward_gen_encoder = GENet(in_dim, num_layers, hidden, pool_type=pool_type, **kwargs)
        
        self.fuse_type = fuse_type
        if fuse_type == FuseType.CONCAT:
            self.pred_head = nn.Sequential(
                nn.Linear(2 * hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, 1),
                nn.Sigmoid(),
            )
        elif fuse_type == FuseType.COSINE:
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
        embed1, embed2 = self.reward_gen_encoder(data1, data2)

        if self.fuse_type == FuseType.ADD:
            pair_embed = embed1 + embed2
        elif self.fuse_type == FuseType.MULTIPLY:
            pair_embed = embed1 * embed2
        elif self.fuse_type == FuseType.CONCAT:
            pair_embed = torch.cat((embed1, embed2), dim=1)
        elif self.fuse_type == FuseType.ABS_DIFF:
            pair_embed = torch.abs(embed1 - embed2)
        elif self.fuse_type == FuseType.COSINE:
            embed1, embed2 = self.pred_head(embed1), self.pred_head(embed2)
            prob = (1.0 + self.cos(embed1, embed2)) / 2.0
            return prob
        
        prob = self.pred_head(pair_embed)
        return prob
