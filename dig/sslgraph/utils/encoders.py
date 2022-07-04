from functools import partial
import torch
import torch.nn.functional as F

from torch.nn import Parameter
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_scatter import scatter_add
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros


class Encoder(torch.nn.Module):
    r"""A wrapped :class:`torch.nn.Module` class for the convinient instantiation of 
    pre-implemented graph encoders.
    
    Args:
        feat_dim (int): The dimension of input node features.
        hidden_dim (int): The dimension of node-level (local) embeddings. 
        n_layer (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
        pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
            (default: :obj:`sum`)
        gnn (string, optional): The type of GNN layer, :obj:`gcn`, :obj:`gin` or 
            :obj:`resgcn`. (default: :obj:`gin`)
        bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
        act (string, optional): The activation function, :obj:`relu` or :obj:`prelu`.
            (default: :obj:`relu`)
        bias (bool, optional): Whether to include bias term in Linear. (default: :obj:`True`)
        xavier (bool, optional): Whether to apply xavier initialization. (default: :obj:`True`)
        node_level (bool, optional): If :obj:`True`, the encoder will output node level
            embedding (local representations). (default: :obj:`False`)
        graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
            embeddings (global representations). (default: :obj:`True`)
        edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
            compute the aggregation. (default: :obj:`False`)
            
    Note
    ----
    For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be 
    :obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`. 
    For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.
            
    Examples
    --------
    >>> feat_dim = dataset[0].x.shape[1]
    >>> encoder = Encoder(feat_dim, 128, n_layer=3, gnn="gin")
    >>> encoder(some_batched_data).shape # graph-level embedding of shape [batch_size, 128*3]
    torch.Size([32, 384]) 
    
    >>> encoder = Encoder(feat_dim, 128, n_layer=5, node_level=True, graph_level=False)
    >>> encoder(some_batched_data).shape # node-level embedding of shape [n_nodes, 128]
    torch.Size([707, 128]) 
    
    >>> encoder = Encoder(feat_dim, 128, n_layer=5, node_level=True, graph_level=False)
    >>> encoder(some_batched_data) # a tuple of graph-level and node-level embeddings
    (tensor([...]), tensor([...])) 
    """
    def __init__(self, feat_dim, hidden_dim, n_layers=5, pool='sum', 
                 gnn='gin', node_level=False, graph_level=True, **kwargs):
        super(Encoder, self).__init__()

        if gnn == 'gin':
            self.encoder = GIN(feat_dim, hidden_dim, n_layers, pool, **kwargs)
        elif gnn == 'gcn':
            self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, **kwargs)
        elif gnn == 'resgcn':
            self.encoder = ResGCN(feat_dim, hidden_dim, num_conv_layers=n_layers, 
                                  global_pool=pool)

        self.node_level = node_level
        self.graph_level = graph_level

    def forward(self, data):
        z_g, z_n = self.encoder(data)
        if self.node_level and self.graph_level:
            return z_g, z_n
        elif self.graph_level:
            return z_g
        else:
            return z_n


class GIN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False, 
                 act='relu', bias=True, xavier=True):
        super(GIN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        self.act = torch.nn.PReLU() if act == 'prelu' else torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            nn = Sequential(Linear(start_dim, hidden_dim, bias=bias),
                            self.act,
                            Linear(hidden_dim, hidden_dim, bias=bias))
            if xavier:
                self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep, x


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False, 
                 act='relu', bias=True, xavier=True, edge_weight=False):
        super(GCN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool
        self.edge_weight = edge_weight
        self.normalize = not edge_weight
        self.add_self_loops = not edge_weight

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias,
                           add_self_loops=self.add_self_loops,
                           normalize=self.normalize)
            if xavier:
                self.weights_init(conv.lin)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_weight:
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep, x
    


class ResGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 improved=False, cached=False, bias=True, edge_norm=True, gfn=False):
        super(ResGCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weights_init()

    def weights_init(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        row, col = edge_index
        
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = ResGCNConv.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    
    

class ResGCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, 
                 num_feat_layers=1, 
                 num_conv_layers=3,
                 num_fc_layers=2, xg_dim=None, bn=True,
                 gfn=False, collapse=False, residual=False,
                 global_pool="sum", dropout=0, edge_norm=True):

        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.collapse = collapse
        self.bn = bn

        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(ResGCNConv, edge_norm=edge_norm, gfn=gfn)

        if xg_dim is not None:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(xg_dim)
            self.lin1_xg = Linear(xg_dim, hidden_dim)
            self.bn2_xg = BatchNorm1d(hidden_dim)
            self.lin2_xg = Linear(hidden_dim, hidden_dim)
        else:
            self.use_xg = False

        if collapse:
            self.bn_feat = BatchNorm1d(feat_dim)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(feat_dim, feat_dim),
                    torch.nn.ReLU(),
                    Linear(feat_dim, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden_dim))
                hidden_in = hidden_dim
        else:
            self.bn_feat = BatchNorm1d(feat_dim)
            feat_gfn = True  # set true so GCNConv is feat transform
            self.conv_feat = ResGCNConv(feat_dim, hidden_dim, gfn=feat_gfn)
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    Linear(hidden_dim, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden_dim))
                self.convs.append(GConv(hidden_dim, hidden_dim))
            self.bn_hidden = BatchNorm1d(hidden_dim)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_dim))
                self.lins.append(Linear(hidden_dim, hidden_dim))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is of shape [n_graphs, feat_dim]
            xg = self.bn1_xg(data.xg) if self.bn else xg
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg) if self.bn else xg
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None
        
        x = self.bn_feat(x) if self.bn else x
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x) if self.bn else x
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        local_rep = x
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x) if self.bn else x
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x, local_rep