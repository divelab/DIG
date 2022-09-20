"""
This is an implementation of ProNet model

"""

from torch_geometric.nn import inits, MessagePassing
from torch_geometric.nn import radius_graph

from .features import d_angle_emb, d_theta_phi_emb

from torch_scatter import scatter
from torch_sparse import matmul

import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F

import numpy as np


num_aa_type = 26
num_side_chain_embs = 8
num_bb_embs = 6

def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):
    """
        A linear method encapsulation similar to PyG's

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
        bias (int)
        weight_initializer (string): (glorot or zeros)
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):
    """
        A layer with two linear modules

        Parameters
        ----------
        in_channels (int)
        middle_channels (int)
        out_channels (int)
        bias (bool)
        act (bool)
    """

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(TwoLinear, self).__init__()
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


class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            output_channels,
            num_radial,
            num_spherical,
            num_layers,
            mid_emb,
            act=swish,
            num_pos_emb=16,
            dropout=0,
            level='allatom'
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        
        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin_feature0 = TwoLinear(num_radial * num_spherical ** 2, mid_emb, hidden_channels)
        if level == 'aminoacid':
            self.lin_feature1 = TwoLinear(num_radial * num_spherical, mid_emb, hidden_channels)
        elif level == 'backbone' or level == 'allatom':
            self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(3*hidden_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()


    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))
        
        feature0 = self.lin_feature0(feature0)
        h0 = self.conv0(x_lin_1, edge_index, feature0)
        h0 = self.lin0(h0)
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)

        h = torch.cat((h0, h1, h2),1)
        for lin in self.lins_cat:
            h = self.act(lin(h)) 

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h)) 
        h = self.final(h)
        return h


class ProNet(nn.Module):
    r"""
         The ProNet from the "Learning Protein Representations via Complete 3D Graph Networks" paper.
        
        Args:
            level: (str, optional): The level of protein representations. It could be :obj:`aminoacid`, obj:`backbone`, and :obj:`allatom`. (default: :obj:`aminoacid`)
            num_blocks (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            mid_emb (int, optional): Embedding size used for geometric features. (default: :obj:`64`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`)
            max_num_neighbors (int, optional): Max number of neighbors during graph construction. (default: :obj:`32`)
            int_emb_layers (int, optional): Number of embedding layers in the interaction block. (default: :obj:`3`)
            out_layers (int, optional): Number of layers for features after interaction blocks. (default: :obj:`2`)
            num_pos_emb (int, optional): Number of positional embeddings. (default: :obj:`16`)
            dropout (float, optional): Dropout. (default: :obj:`0`)
            data_augment_eachlayer (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to the node features before each interaction block. (default: :obj:`False`)
            euler_noise (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to Euler angles. (default: :obj:`False`)
            
    """
    def __init__(
            self,
            level='aminoacid',
            num_blocks=4,
            hidden_channels=128,
            out_channels=1,
            mid_emb=64,
            num_radial=6,
            num_spherical=2,
            cutoff=10.0,
            max_num_neighbors=32,
            int_emb_layers=3,
            out_layers=2,
            num_pos_emb=16,
            dropout=0,
            data_augment_eachlayer=False,
            euler_noise = False,
    ):
        super(ProNet, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.act = swish

        self.feature0 = d_theta_phi_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature1 = d_angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        if level == 'aminoacid':
            self.embedding = Embedding(num_aa_type, hidden_channels)
        elif level == 'backbone':
            self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs, hidden_channels)
        elif level == 'allatom':
            self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs + num_side_chain_embs, hidden_channels)
        else:
            print('No supported model!')

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_layers=int_emb_layers,
                    mid_emb=mid_emb,
                    act=self.act,
                    num_pos_emb=num_pos_emb,
                    dropout=dropout,
                    level=level
                )
                for _ in range(num_blocks)
            ]
        )

        self.lins_out = torch.nn.ModuleList()
        for _ in range(out_layers-1):
            self.lins_out.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self, edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, batch_data):

        z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs

        device = z.device

        if self.level == 'aminoacid':
            x = self.embedding(z)
        elif self.level == 'backbone':
            x = torch.cat([torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs], dim = 1)
            x = self.embedding(x)
        elif self.level == 'allatom':
            x = torch.cat([torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs, side_chain_embs], dim = 1)
            x = self.embedding(x)
        else:
            print('No supported model!')

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)
        j, i = edge_index

        # Calculate distances.
        dist = (pos[i] - pos[j]).norm(dim=1)
        
        num_nodes = len(z)

        # Calculate angles theta and phi.
        refi0 = (i-1)%num_nodes
        refi1 = (i+1)%num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
        theta = torch.atan2(b, a)

        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a)

        feature0 = self.feature0(dist, theta, phi)

        if self.level == 'backbone' or self.level == 'allatom':
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7
            
            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z)
            
            angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1)/Or1_z_length, (Or1_x * Or1_Or2_N).sum(dim=-1))
            angle2 = torch.atan2(torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
            angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1)/Or2_z_length, (Or1_Or2_N * Or2_x).sum(dim=-1))

            if self.euler_noise:
                euler_noise = torch.clip(torch.empty(3,len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]

            feature1 = torch.cat((self.feature1(dist, angle1), self.feature1(dist, angle2), self.feature1(dist, angle3)),1)

        elif self.level == 'aminoacid':
            refi = (i-1)%num_nodes

            refj0 = (j-1)%num_nodes
            refj = (j-1)%num_nodes
            refj1 = (j+1)%num_nodes

            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]

            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i])
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j])
            a = (plane1 * plane2).sum(dim=-1) 
            b = (torch.cross(plane1, plane2) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)

            feature1 = self.feature1(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            if self.data_augment_eachlayer:
                # add gaussian noise to features
                gaussian_noise = torch.clip(torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                x += gaussian_noise
            x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)

        y = scatter(x, batch, dim=0)

        for lin in self.lins_out:
            y = self.relu(lin(y))
            y = self.dropout(y)        
        y = self.lin_out(y)
        return y

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
