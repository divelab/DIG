import sys, torch
import torch.nn as nn
from .contrastive import Contrastive
from sslgraph.contrastive.views_fn import node_attr_mask, edge_perturbation, uniform_sample, RW_sample, random_view

class GraphCL(Contrastive):
    
    def __init__(self, dim, aug_1, aug_2, aug_ratio=0.2, device=None,
                 choice_model='last', model_path='models'):
        '''
        dim: Integer. Embedding dimension.
        aug1, aug2: String. Should be in ['dropN', 'permE', 'subgraph', 
                    'maskN', 'random2', 'random3', 'random4'].
        aug_ratio: Float between (0,1).
        '''
        views_fn = []
        
        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(uniform_sample(ratio=aug_ratio))
            elif aug == 'permE':
                views_fn.append(edge_perturbation(ratio=aug_ratio))
            elif aug == 'subgraph':
                views_fn.append(RW_sample(ratio=aug_ratio))
            elif aug == 'maskN':
                views_fn.append(node_attr_mask(mask_ratio=aug_ratio))
            elif aug == 'random2':
                canditates = [uniform_sample(ratio=aug_ratio),
                              RW_sample(ratio=aug_ratio)]
                views_fn.append(random_view(canditates))
            elif aug == 'random4':
                canditates = [uniform_sample(ratio=aug_ratio),
                              RW_sample(ratio=aug_ratio),
                              edge_perturbation(ratio=aug_ratio)]
                views_fn.append(random_view(canditates))
            elif aug == 'random3':
                canditates = [uniform_sample(ratio=aug_ratio),
                              RW_sample(ratio=aug_ratio),
                              edge_perturbation(ratio=aug_ratio),
                              node_attr_mask(mask_ratio=aug_ratio)]
                views_fn.append(random_view(canditates))
            else:
                raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                'maskN', 'random2', 'random3', 'random4'] or None.")
        
        super(GraphCL, self).__init__(objective='NCE',
                                      views_fn=views_fn,
                                      z_dim=dim,
                                      proj='MLP',
                                      node_level=False,
                                      choice_model=choice_model,
                                      model_path=model_path,
                                      device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, proj in super(GraphCL, self).train(encoders, data_loader, 
                                                    optimizer, epochs, per_epoch_out):
            yield enc