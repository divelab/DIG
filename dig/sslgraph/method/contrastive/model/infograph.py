import torch.nn as nn
from .contrastive import Contrastive


class ProjHead(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    
    
class InfoG_enc(nn.Module):
    def __init__(self, encoder, z_g_dim, z_n_dim):
        
        super(InfoG_enc, self).__init__()
        self.fc = nn.Linear(z_g_dim, z_n_dim)
        self.encoder = encoder
        
    def forward(self, data):
        zg, zn = self.encoder(data)
        zg = self.fc(zg)
        return zg


class InfoGraph(Contrastive):
    r"""
    Contrastive learning method proposed in the paper `InfoGraph: Unsupervised and 
    Semi-supervised Graph-Level Representation Learning via Mutual Information 
    Maximization <https://arxiv.org/abs/1908.01000>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_infograph.ipynb>`_ 
    for an example of usage.

    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`InfoGraph`.
    
    Args:
        g_dim (int): The embedding dimension for graph-level (global) representations.
        n_dim (int): The embedding dimension for node-level (local) representations. Typically,
            when jumping knowledge is included in the encoder, we have 
            :obj:`g_dim` = :obj:`n_layers` * :obj:`n_dim`.
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, g_dim, n_dim, **kwargs):

        views_fn = [lambda x: x]
        proj = ProjHead(g_dim, n_dim)
        proj_n = ProjHead(n_dim, n_dim)
        super(InfoGraph, self).__init__(objective='JSE',
                                        views_fn=views_fn,
                                        node_level=True,
                                        z_dim=g_dim,
                                        z_n_dim=n_dim,
                                        proj=proj,
                                        proj_n=proj_n,
                                        **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        for enc, (proj, proj_n) in super(InfoGraph, self).train(encoders, data_loader, 
                                                                optimizer, epochs, per_epoch_out):
            yield InfoG_enc(enc, self.z_dim, self.z_n_dim)
