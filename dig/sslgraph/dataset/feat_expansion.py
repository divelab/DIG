import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, remove_self_loops, add_self_loops

class FeatureExpander(MessagePassing):
    """
    """
    def __init__(self, degree=True, onehot_maxdeg=0, AK=1, centrality=False,
                remove_edges="none", edge_noises_add=0, edge_noises_delete=0, 
                group_degree=0):
        super(FeatureExpander, self).__init__('add', 'source_to_target')

        self.degree = degree
        self.onehot_maxdeg = onehot_maxdeg
        self.AK = AK
        self.centrality = centrality
        self.remove_edges = remove_edges
        self.edge_noises_add = edge_noises_add
        self.edge_noises_delete = edge_noises_delete
        self.group_degree = group_degree

        # edge norm is used, and set A diag to it
        self.edge_norm_diag = 1e-8

    def transform(self, data):
        if data.x is None:
            data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        
        # ignore edge noises

        deg, deg_onehot = self.compute_degree(data.edge_index, data.num_nodes)
        akx = self.compute_akx(data.num_nodes, data.x, data.edge_index)
        cent = self.compute_centrality(data)
        # data.x = torch.cat([data.x, deg, deg_onehot, akx, cent], dim=-1)
        # data.x = torch.cat([data.x, deg_onehot, akx, cent, deg], dim=-1)
        data.x = torch.cat([deg, data.x, deg_onehot, akx, cent], dim=-1)

        return data
    
    def compute_degree(self, edge_index, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes)
        deg = deg.view((-1, 1))

        if self.onehot_maxdeg is not None and self.onehot_maxdeg > 0:
            max_deg = torch.tensor(self.onehot_maxdeg, dtype=deg.dtype)
            deg_capped = torch.min(deg, max_deg).type(torch.int64)
            deg_onehot = nn.functional.one_hot(deg_capped.view(-1), num_classes=self.onehot_maxdeg + 1)
            deg_onehot = deg_onehot.type(deg.dtype)
        else:
            deg_onehot = self.empty_feature(num_nodes)

        if not self.degree:
            deg = self.empty_feature(num_nodes)
        
        return deg, deg_onehot

    
    def compute_akx(self, num_nodes, x, edge_index, edge_weight=None):
        if self.AK is None or self.AK <= 0:
            return self.empty_feature(num_nodes)
    
    def compute_centrality(self, data):
        if not self.centrality:
            return self.empty_feature(data.num_nodes)
    
    def empty_feature(self, num_nodes):
        return torch.zeros([num_nodes, 0])
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, diag_val=1e-8, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),
                                 diag_val,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class CatDegOnehot(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg
            
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_degree)
    
    
def get_max_deg(dataset):
    max_deg = 0
    for data in dataset:
        row, col = data.edge_index
        num_nodes = data.num_nodes
        deg = degree(row, num_nodes)
        deg = max(deg).item()
        if deg > max_deg:
            max_deg = int(deg)
    return max_deg