import sys, torch
import torch.nn as nn
from .contrastive import Contrastive
from sslgraph.contrastive.views_fn import node_attr_mask, edge_perturbation, combine

class GRACE(Contrastive):
    
    def __init__(self, dim, dropE_rate_1, dropE_rate_2, maskN_rate_1, maskN_rate_2, 
                 tau=0.5, device=None):
        '''
        dim: Integer. Embedding dimension.
        dropE_rate_1, dropE_rate_2, maskN_rate_1, maskN_rate_2: Float in [0, 1).
        '''
        view_fn_1 = combine([edge_perturbation(ratio=dropE_rate_1),
                             node_attr_mask(mask_ratio=maskN_rate_1)])
        view_fn_2 = combine([edge_perturbation(ratio=dropE_rate_2),
                             node_attr_mask(mask_ratio=maskN_rate_2)])
        views_fn = [view_fn_1, view_fn_2]
        
        super(GRACE, self).__init__(objective='NCE',
                                    views_fn=views_fn,
                                    graph_level=False,
                                    node_level=True,
                                    z_n_dim=dim,
                                    proj_n='MLP',
                                    tau=tau,
                                    device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GRACE removes projection heads after pre-training
        for enc, proj in super().train(encoders, data_loader, 
                                       optimizer, epochs, per_epoch_out):
            yield enc