import sys, torch
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
    
    def __init__(self, z_g_dim, z_n_dim, device=None):
        '''
        Args:
            diffusion_type: String. Diffusion instantiation mode with two options:
                            'ppr': Personalized PageRank
                            'heat': heat kernel
            alpha: Float in (0,1). Teleport probability in a random walk.
            t: Integer. Diffusion time.
        '''
        views_fn = [lambda x: x]
        proj = ProjHead(z_g_dim, z_n_dim)
        proj_n = ProjHead(z_n_dim, z_n_dim)
        super(InfoGraph, self).__init__(objective='JSE',
                                        views_fn=views_fn,
                                        node_level=True,
                                        z_dim=z_g_dim,
                                        z_n_dim=z_n_dim,
                                        proj=proj,
                                        proj_n=proj_n,
                                        device=device)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        for enc, (proj, proj_n) in super(InfoGraph, self).train(encoders, data_loader, 
                                                                optimizer, epochs, per_epoch_out):
            yield InfoG_enc(enc, self.z_dim, self.z_n_dim)