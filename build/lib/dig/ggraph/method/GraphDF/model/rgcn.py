import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationGraphConvolution(nn.Module):
    """
    Relation GCN layer. 
    """

    def __init__(self, in_features, out_features, edge_dim=3, aggregate='sum', dropout=0., use_relu=True, bias=False):
        '''
        :param in/out_features: scalar of channels for node embedding
        :param edge_dim: dim of edge type, virtual type not included
        '''
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.aggregate = aggregate
        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = None

        self.weight = nn.Parameter(torch.FloatTensor(
            self.edge_dim, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(
                self.edge_dim, 1, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        typically d=9 e=3
        :return:
        updated x with shape (batch, N, d)
        '''
        x = F.dropout(x, p=self.dropout, training=self.training)  # (b, N, d)

        batch_size = x.size(0)

        # transform
        support = torch.einsum('bid, edh-> beih', x, self.weight)
        output = torch.einsum('beij, bejh-> beih', adj,
                              support)  # (batch, e, N, d)

        if self.bias is not None:
            output += self.bias
        if self.act is not None:
            output = self.act(output)  # (b, E, N, d)
        output = output.view(batch_size, self.edge_dim, x.size(
            1), self.out_features)  # (b, E, N, d)

        if self.aggregate == 'sum':
            # sum pooling #(b, N, d)
            node_embedding = torch.sum(output, dim=1, keepdim=False)
        elif self.aggregate == 'max':
            # max pooling  #(b, N, d)
            node_embedding = torch.max(output, dim=1, keepdim=False)
        elif self.aggregate == 'mean':
            # mean pooling #(b, N, d)
            node_embedding = torch.mean(output, dim=1, keepdim=False)
        elif self.aggregate == 'concat':
            #! implementation wrong
            node_embedding = torch.cat(torch.split(
                output, dim=1, split_size_or_sections=1), dim=3)  # (b, 1, n, d*e)
            node_embedding = torch.squeeze(
                node_embedding, dim=1)  # (b, n, d*e)
        else:
            print('GCN aggregate error!')
        return node_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class RGCN(nn.Module):
    def __init__(self, nfeat, nhid=128, nout=128, edge_dim=3, num_layers=3, dropout=0., normalization=False):
        '''
        :num_layars: the number of layers in each R-GCN
        '''
        super(RGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nout = nout
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.dropout = dropout
        self.normalization = normalization

        self.emb = nn.Linear(nfeat, nfeat, bias=False) 

        self.gc1 = RelationGraphConvolution(
            nfeat, nhid, edge_dim=self.edge_dim, aggregate='sum', use_relu=True, dropout=self.dropout, bias=False)

        self.gc2 = nn.ModuleList([RelationGraphConvolution(nhid, nhid, edge_dim=self.edge_dim, aggregate='sum',
                                                           use_relu=True, dropout=self.dropout, bias=False)
                                  for i in range(self.num_layers-2)])

        self.gc3 = RelationGraphConvolution(
            nhid, nout, edge_dim=self.edge_dim, aggregate='sum', use_relu=False, dropout=self.dropout, bias=False)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        :return:
        '''
        # embedding layer
        x = self.emb(x)

        # first GCN layer
        x = self.gc1(x, adj)

        # hidden GCN layer(s)
        for i in range(self.num_layers-2):
            x = self.gc2[i](x, adj)  # (#node, #class)

        # last GCN layer
        x = self.gc3(x, adj)  # (batch, N, d)

        # return node embedding
        return x