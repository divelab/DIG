import sys, torch
import torch.nn as nn
from .contrastive import Contrastive
from sslgraph.contrastive.views_fn import diffusion, diffusion_with_sample


class MVGRL_enc(nn.Module):
    '''
        MVGRL includes projection heads and combines two views and encoders
        when inferencing graph-level representation.
    '''
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
    
    def __init__(self, z_dim, z_n_dim, diffusion_type='ppr', alpha=0.2, t=5, 
                 graph_level_output=True, node_level_output=False, device=None, 
                 choice_model='best', model_path='models'):
        '''
        Args:
            diffusion_type: String. Diffusion instantiation mode with two options:
                'ppr': Personalized PageRank
                'heat': heat kernel
            alpha: Float in (0,1). Teleport probability in a random walk.
            t: Integer. Diffusion time.
            subgraph: Boolean. Whether to sample subgraph from a large graph. 
                Set to True for node-level tasks on large graphs.
        '''
        self.views_fn = [lambda x: x,
                         diffusion(mode=diffusion_type, alpha=alpha, t=t)]
        self.graph_level = graph_level_output
        self.node_level = node_level_output
        super(MVGRL, self).__init__(objective='JSE',
                                    views_fn=self.views_fn,
                                    node_level=True,
                                    z_dim=z_dim,
                                    z_n_dim=z_n_dim,
                                    proj=ProjHead(z_dim, z_n_dim),
                                    proj_n=ProjHead(z_n_dim, z_n_dim),
                                    choice_model=choice_model,
                                    model_path=model_path,
                                    device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        
        for encs, (proj, proj_n) in super(MVGRL, self).train(encoders, data_loader, 
                                                             optimizer, epochs, per_epoch_out):
            encoder = MVGRL_enc(encs[0], encs[1], proj, proj_n, 
                                self.views_fn, True, False)
            yield encoder   
    
    
    
class NodeMVGRL(Contrastive):
    
    def __init__(self, z_dim, z_n_dim, diffusion_type='ppr', alpha=0.2, t=5, 
                 batch_size=2, num_nodes=2000,
                 graph_level_output=False, node_level_output=True, device=None, 
                 choice_model='best', model_path='models'):
        '''
        Args:
            diffusion_type: String. Diffusion instantiation mode with two options:
                'ppr': Personalized PageRank
                'heat': heat kernel
            alpha: Float in (0,1). Teleport probability in a random walk.
            t: Integer. Diffusion time.
            subgraph: Boolean. Whether to sample subgraph from a large graph. 
                Set to True for node-level tasks on large graphs.
        '''
        self.graph_level = graph_level_output
        self.node_level = node_level_output
        self.mode = diffusion_type
        self.alpha = alpha
        self.t = t
        views_fn = [diffusion_with_sample(num_nodes, batch_size, mode=self.mode,
                                          alpha=self.alpha, t=self.t), None]
        
        super(NodeMVGRL, self).__init__(objective='JSE',
                                        views_fn=views_fn,
                                        node_level=True,
                                        z_dim=z_dim,
                                        z_n_dim=z_n_dim,
                                        proj=nn.Sigmoid(),
                                        proj_n='linear',
                                        neg_by_crpt=True,
                                        choice_model=choice_model,
                                        model_path=model_path,
                                        device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        
        for encs, (proj, proj_n) in super(NodeMVGRL, self).train(encoders, data_loader, 
                                                             optimizer, epochs, per_epoch_out):
            views_fn = [lambda x: x,
                        diffusion(mode=self.mode, alpha=self.alpha, t=self.t)]
            # mvgrl for node-level tasks follows DGI, excluding the projection heads after pretraining
            mvgrl_enc = MVGRL_enc(encs[0], encs[1], (lambda x: x), (lambda x: x), 
                                  views_fn, False, True)
            yield mvgrl_enc