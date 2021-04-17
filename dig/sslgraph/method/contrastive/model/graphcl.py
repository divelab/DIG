import sys, torch
import torch.nn as nn
from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import node_attr_mask, edge_perturbation, \
    uniform_sample, RW_sample, random_view

class GraphCL(Contrastive):
    r"""    
    Contrastive learning method proposed in the paper `Graph Contrastive Learning with 
    Augmentations <https://arxiv.org/abs/2010.13902>`_.

    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GraphCL`.
    
    Args:
        dim (int): The embedding dimension.
        aug1 (sting, optinal): Types of augmentation for the first view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug2 (sting, optinal): Types of augmentation for the second view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug_ratio (float, optional): The ratio of augmentations. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, dim, aug_1=None, aug_2=None, aug_ratio=0.2, **kwargs):
        
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
                                      **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, proj in super(GraphCL, self).train(encoders, data_loader, 
                                                    optimizer, epochs, per_epoch_out):
            yield enc