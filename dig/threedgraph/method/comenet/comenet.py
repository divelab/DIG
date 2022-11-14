from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits

from .features import angle_emb, torsion_emb

from torch_scatter import scatter, scatter_min

from torch.nn import Embedding

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt

try:
    import sympy as sym
except ImportError:
    sym = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)

class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
    ):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            middle_channels,
            num_radial,
            num_spherical,
            num_layers,
            output_channels,
            act=swish
    ):
        super(SimpleInteractionBlock, self).__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, middle_channels, hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, middle_channels, hidden_channels)

        # Dense transformations of input messages.
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.lin(x))

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))

        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        return h


class ComENet(nn.Module):
    r"""
         The ComENet from the `"ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs" <https://arxiv.org/abs/2206.08515>`_ paper.
        
        Args:
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`8.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`256`)
            middle_channels (int, optional): Middle embedding size for the two layer linear block. (default: :obj:`256`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`3`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
    """
    def __init__(
            self,
            cutoff=8.0,
            num_layers=4,
            hidden_channels=256,
            middle_channels=64,
            out_channels=1,
            num_radial=3,
            num_spherical=2,
            num_output_layers=3,
    ):
        super(ComENet, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act

        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.emb = EmbeddingBlock(hidden_channels, act)

        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def _forward(self, data):
        batch = data.batch
        z = data.z.long()
        pos = data.pos
        num_nodes = z.size(0)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        # Embedding block.
        x = self.emb(z)

        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        for lin in self.lins:
            x = self.act(lin(x))
        x = self.lin_out(x)

        energy = scatter(x, batch, dim=0)
        return energy

    def forward(self, batch_data):
        return self._forward(batch_data)