import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, activation = F.relu, dropout = None, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.init_params()

    def init_params(self):
        if self.W is not None:
            init.xavier_uniform_(self.W)
        if self.b is not None:
            init.zeros_(self.b)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W

        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x
    
class GCN_Body(nn.Module):
    r'''
    Implementation of GNNs used for generating contrastive loss between original graph and it's fair representation.
    '''
    def __init__(self, in_feats, n_hidden, out_feats, dropout, nlayer):
        super(GCN_Body, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden))
        # hidden layers
        for i in range(nlayer - 2):
            self.layers.append(GCNLayer(n_hidden, n_hidden))
        # output layer
        self.layers.append(GCNLayer(n_hidden, out_feats))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = x
        cnt = 0
        for layer in self.layers:
            if self.dropout and cnt != 0:
                h = self.dropout(h)
            cnt += 1
            h = (layer(g, h))
        return h

class GCN(nn.Module):
    r'''
    Implemention of GCN used in adversarial model k: (A',X') -> S_hat
    '''
    def __init__(self, in_feats, n_hidden, out_feats, nclass, dropout = 0.2, nlayer = 2):
        super(GCN, self).__init__()
        self.body = GCN_Body(in_feats, n_hidden, out_feats, dropout, nlayer)
        self.fc = nn.Sequential(
                nn.Linear(out_feats, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, nclass),
                )

    def forward(self, g, x):
        h = self.body(g, x)
        x = self.fc(h)
        return x , h
