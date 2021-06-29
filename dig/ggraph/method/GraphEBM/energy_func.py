import torch
from torch import nn
from torch.nn import functional as F



def swish(x):
    return x * torch.sigmoid(x)



class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module



class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_edge_type, std, bound=True, add_self=False):
        super(GraphConv, self).__init__()
        
        self.add_self = add_self
        if self.add_self:
            self.linear_node = spectral_norm(nn.Linear(in_channels, out_channels), std=std, bound=bound)
        self.linear_edge = spectral_norm(nn.Linear(in_channels, out_channels * num_edge_type), std=std, bound=bound)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
        mb, node, _ = h.shape 
        if self.add_self:
            h_node = self.linear_node(h) 
        m = self.linear_edge(h)
        m = m.reshape(mb, node, self.out_ch, self.num_edge_type) 
        m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
        hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
        hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)
        if self.add_self:
            return hr+h_node  #
        else:
            return hr
    
    
    


class EnergyFunc(nn.Module):

    def __init__(self, n_atom_type, hidden, num_edge_type=4, swish=True, depth=2, add_self=False, dropout=0):
        super(EnergyFunc, self).__init__()

        self.depth = depth
        self.graphconv1 = GraphConv(n_atom_type, hidden, num_edge_type, std=1, bound=False, add_self=add_self)
        self.graphconv = nn.ModuleList(GraphConv(hidden, hidden, num_edge_type, std=1e-10, add_self=add_self) for i in range(self.depth))
        self.swish = swish
        self.dropout = dropout
        self.linear = nn.Linear(hidden, 1)
            
        
    def forward(self, adj, h):
        h = h.permute(0, 2, 1)
        out = self.graphconv1(adj, h)
            
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        if self.swish:
            out = swish(out)
        else:
            out = F.leaky_relu(out, negative_slope=0.2)


        for i in range(self.depth):
            out = self.graphconv[i](adj, out)
                
            out = F.dropout(out, p=self.dropout, training=self.training)
            if self.swish:
                out = swish(out)
            else:
                out = F.relu(out)
        
        out = out.sum(1) # (batchsize, node, ch) --> (batchsize, ch)
        out = self.linear(out)
        
        return out # Energy value (batchsize, 1)