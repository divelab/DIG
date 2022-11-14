import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
import torch.nn.functional as F
from math import sqrt

from .geometric_computing import xyztodat
from .features import dist_emb, angle_emb, torsion_emb

try:
    import sympy as sym
except ImportError:
    sym = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)

class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_node_types, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(num_node_types, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return (e1, e2), x


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, 
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
    
    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        
        x1,x2 = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf
        x_kj = self.act(self.lin_down(x_kj))
        
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf
        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))
        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1
        
        non_iso_idx = torch.cat((idx_ji, idx_kj))
        e1 = x1 + scatter(e1[non_iso_idx] - x1[non_iso_idx], non_iso_idx, dim=0, dim_size=x1.size(0), reduce='mean')
        e2 = x2 + scatter(e2[non_iso_idx] - x2[non_iso_idx], non_iso_idx, dim=0, dim_size=x2.size(0), reduce='mean')
        
        return e1, e2


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, num_output_layers, act):
        super(update_v, self).__init__()
        self.act = act

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers-1):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)

    def forward(self, e, i, num_nodes):
        _, e2 = e
        v = scatter(e2, i, dim=0, dim_size=num_nodes)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        v = scatter(v[i], i, dim=0, dim_size=num_nodes, reduce='mean')
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


class SphereNet(torch.nn.Module):
    def __init__(
        self, cutoff, num_node_types, num_layers, 
        hidden_channels, int_emb_size, basis_emb_size, out_emb_channels,
        num_spherical, num_radial, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, 
        act=swish):

        super(SphereNet, self).__init__()

        self.cutoff = cutoff

        self.init_e = init(num_node_types, num_radial, hidden_channels, act)
        self.init_v = update_v(hidden_channels, out_emb_channels, num_output_layers, act)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        
        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, num_output_layers, act) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, num_before_skip, num_after_skip,act) for _ in range(num_layers)])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
    
    def dist_only_forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        emb = self.emb.dist_emb(dist)

        e, _ = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, z.size(0))
        # u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)

        for l in range(len(self.update_es)-1):
            v = self.update_vs[l](e, i, z.size(0))
            # u = self.update_us[l](u, v, batch)
        
        v = self.update_vs[-1](e, i, z.size(0))
        # u = self.update_us[-1](u, v, batch)

        return v

    def forward(self, z, pos, batch):
        # z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes=z.size(0)
        
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyztodat(pos, edge_index, num_nodes, batch)

        emb = self.emb(dist, angle, torsion, idx_kj)

        #Initialize edge, node, graph features
        init_e, node_type_emb = self.init_e(z, emb[0], i, j)

        e = init_e
        v = self.init_v(e, i, num_nodes)
        # u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)

        for l in range(len(self.update_es) - 1):
            e = self.update_es[l](e, emb, idx_kj, idx_ji)
            v = self.update_vs[l](e, i, num_nodes)
            # u = self.update_us[l](u, v, batch) #u += scatter(v, batch, dim=0)
        
        e = self.update_es[-1](e, emb, idx_kj, idx_ji)
        v = self.update_vs[-1](e, i, num_nodes)
        v = node_type_emb + scatter(v[j]-node_type_emb[j], j, dim=0, reduce='mean', dim_size=num_nodes)
        # u = self.update_us[-1](u, v, batch)

        return v