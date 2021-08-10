import torch.nn as nn
from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import Diffusion, DiffusionWithSample


class MVGRL_enc(nn.Module):
    # MVGRL includes projection heads and combines two views and 
    # encoders when inferencing graph-level representation.

    def __init__(self, encoder_0, encoder_1, 
                 proj, proj_n, views_fn, 
                 graph_level=True, node_level=True):
        
        super(MVGRL_enc, self).__init__()
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        self.proj = proj
        self.proj_n = proj_n
        self.views_fn = views_fn
        self.graph_level = graph_level
        self.node_level = node_level
        
    def forward(self, data):
        device = data.x.device
        
        zg_1, zn_1 = self.encoder_0(self.views_fn[0](data.to('cpu'))
                                    .to(device))
        zg_1 = self.proj(zg_1)
        zn_1 = self.proj_n(zn_1)
        
        zg_2, zn_2 = self.encoder_1(self.views_fn[1](data.to('cpu'))
                                    .to(device))
        zg_2 = self.proj(zg_2)
        zn_2 = self.proj_n(zn_2)
        
        if self.graph_level and self.node_level:
            return (zg_1 + zg_2), (zn_1 + zn_2)
        elif self.graph_level:
            return zg_1 + zg_2
        elif self.node_level:
            return zn_1 + zn_2
        else:
            return None

        
class ProjHead(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(ProjHead, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)

    
class MVGRL(Contrastive):
    r'''
    Contrastive learning method for graph-level tasks proposed in the paper `Contrastive 
    Multi-View Representation Learning on Graphs (MVGRL) <https://arxiv.org/pdf/2006.05582v1.pdf>`_. 
    You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_mvgrl.ipynb>`_ 
    for an example of usage.
    
    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`MVGRL`.
        
    Args:
        g_dim (int): The embedding dimension for graph-level (global) representations.
        n_dim (int): The embedding dimension for node-level (local) representations. Typically,
            when jumping knowledge is included in the encoder, we have 
            :obj:`g_dim` = :obj:`n_layers` * :obj:`n_dim`.
        diffusion_type (string, optional): Diffusion instantiation mode with two options:
            :obj:`"ppr"`: Personalized PageRank. 
            :obj:`"heat"`: heat kernel. 
            (default: :obj:`"ppr"`)
        alpha (float, optional): Teleport probability in a random walk. (default: :obj:`0.2`)
        t (int, optinal): Diffusion time. (default: :obj:`5`)
    '''
    
    def __init__(self, g_dim, n_dim, diffusion_type='ppr', alpha=0.2, t=5, **kwargs):

        self.views_fn = [lambda x: x,
                         Diffusion(mode=diffusion_type, alpha=alpha, t=t)]
        super(MVGRL, self).__init__(objective='JSE',
                                    views_fn=self.views_fn,
                                    node_level=True,
                                    z_dim=g_dim,
                                    z_n_dim=n_dim,
                                    proj=ProjHead(g_dim, n_dim),
                                    proj_n=ProjHead(n_dim, n_dim),
                                    **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        
        for encs, (proj, proj_n) in super(MVGRL, self).train(encoders, data_loader, 
                                                             optimizer, epochs, per_epoch_out):
            encoder = MVGRL_enc(encs[0], encs[1], proj, proj_n, 
                                self.views_fn, True, False)
            yield encoder   
    
    
    
class NodeMVGRL(Contrastive):
    r'''    
    Contrastive learning method for node-level tasks proposed in the paper `Contrastive 
    Multi-View Representation Learning on Graphs (MVGRL) <https://arxiv.org/pdf/2006.05582v1.pdf>`_. 

    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`NodeMVGRL`.

    Args:
        z_g_dim (int): The embedding dimension for graph-level (global) representations.
        z_n_dim (int): The embedding dimension for node-level (local) representations. Typically,
            when jumping knowledge is included in the encoder, we have 
            :obj:`z_g_dim` = :obj:`n_layers` * :obj:`z_n_dim`.
        diffusion_type (string, optional): Diffusion instantiation mode with two options:
            :obj:`"ppr"`: Personalized PageRank. 
            :obj:`"heat"`: heat kernel. 
            (default: :obj:`"ppr"`)
        alpha (float, optional): Teleport probability in a random walk. (default: :obj:`0.2`)
        t (int, optinal): Diffusion time. (default: :obj:`5`)
        batch_size (int, optinal): Number of subgraph samples in each minibatch. (default: :obj:`2`)
        num_nodes (int, optinal): Number of nodes sampled in each subgraph. (default: :obj:`2000`)
    '''
    
    def __init__(self, z_dim, z_n_dim, diffusion_type='ppr', alpha=0.2, t=5, 
                 batch_size=2, num_nodes=2000, **kwargs):

        self.mode = diffusion_type
        self.alpha = alpha
        self.t = t
        views_fn = [DiffusionWithSample(num_nodes, batch_size, mode=self.mode,
                                        alpha=self.alpha, t=self.t), None]
        
        super(NodeMVGRL, self).__init__(objective='JSE',
                                        views_fn=views_fn,
                                        node_level=True,
                                        z_dim=z_dim,
                                        z_n_dim=z_n_dim,
                                        proj='MLP',
                                        proj_n='linear',
                                        neg_by_crpt=True,
                                        **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        
        for encs, (proj, proj_n) in super(NodeMVGRL, self).train(encoders, data_loader, 
                                                             optimizer, epochs, per_epoch_out):
            views_fn = [lambda x: x,
                        Diffusion(mode=self.mode, alpha=self.alpha, t=self.t)]
            # mvgrl for node-level tasks follows DGI, excluding the projection heads after pretraining
            mvgrl_enc = MVGRL_enc(encs[0], encs[1], (lambda x: x), (lambda x: x), 
                                  views_fn, False, True)
            yield mvgrl_enc
