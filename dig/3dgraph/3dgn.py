import os
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import radius_graph
from torch_geometric.nn.acts import swish
from torch_scatter import scatter
from torch_sparse import SparseTensor
import torch.nn.functional as F

from math import sqrt, pi as PI
from spherenet import init_spherenet, emb_spherenet, update_e_spherenet, update_v_spherenet, update_u_spherenet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class threedgn(torch.nn.Module):
    def __init__(
        self, model, cutoff, num_layers, 
        hidden_channels_schnet, num_filters_schnet, num_gaussians,
        hidden_channels, out_channels, int_emb_size, basis_emb_size, out_emb_channels, 
        num_spherical, num_radial, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, 
        act=swish, output_init='GlorotOrthogonal'):
        super(threedgn, self).__init__()

        self.cutoff = cutoff

        if model == 'shperenet':
            self.init_e = init_spherenet(num_radial, hidden_channels, act)
            self.init_v = update_v_spherenet(num_radial, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
            self.emb = emb_spherenet(num_spherical, num_radial, self.cutoff, envelope_exponent)
            self.update_e = update_e_spherenet(hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, num_before_skip, num_after_skip, act=swish)
            self.update_v = update_v_spherenet(num_radial, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
            self.update_u = update_u_spherenet()
        
        self.update_vs = torch.nn.ModuleList([
            self.update_v(num_radial, hidden_channels,
                out_emb_channels,
                out_channels, num_output_layers,
                act, output_init, 
            )
            for _ in range(num_layers)
        ])

        self.update_es = torch.nn.ModuleList([
            self.update_e(
                hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])

        self.update_us = torch.nn.ModuleList([self.update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()


    def xyztodat(self, pos, edge_index, num_nodes):
        j, i = edge_index  # j->i

        # Calculate distances. # number of edges
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        value = torch.arange(j.size(0), device=j.device)
        adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[j]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = i.repeat_interleave(num_triplets)
        idx_j = j.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        # Calculate angles. 0 to pi
        pos_ji = pos[idx_i] - pos[idx_j]
        pos_jk = pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
        b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
        angle = torch.atan2(b, a)
                
        idx_batch = torch.arange(len(idx_i),device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets - 1
        num_triplets_t = num_triplets.repeat_interleave(repeat)
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n       
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

        # Calculate torsions.
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_j0)
        plane2 = torch.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        torsion1 = torch.atan2(b, a) # -pi to pi
        torsion1[torsion1<=0]+=2*PI # 0 to 2pi
        torsion = scatter(torsion1,idx_batch_t,reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji


    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        num_nodes=z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = self.xyztodat(pos, edge_index, num_nodes)

        emb = self.emb(dist, angle, torsion, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, num_nodes=pos.size(0))
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, emb, i)
            u = update_u(u, v, batch)

        return u
