import torch
import torch.nn as nn
from layers import GCN, AvgReadout
from torch.nn import Sequential, Linear, ReLU
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv


class Encoder(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_gc_layers=5, activation='prelu'):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            start_dim = dim_out if i else dim_in
            conv = GCN(start_dim, dim_out, activation)
            bn = torch.nn.BatchNorm1d(dim_out)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, adj, sparse):
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, adj, sparse)
            x = self.bns[i](torch.squeeze(x))
            x = torch.unsqueeze(x, 0)
            xs.append(x)
        concat_enc = torch.cat(xs, 2)

        return x, concat_enc


class Decoder(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_gc_layers=5, activation='prelu'):
        super(Decoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            end_dim = dim_out if i == num_gc_layers - 1 else dim_in
            conv = GCN(dim_in, end_dim, activation)
            bn = torch.nn.BatchNorm1d(end_dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, adj, sparse):
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, adj, sparse)
            if i < self.num_gc_layers - 1:  # no bn for last layer
                x = self.bns[i](torch.squeeze(x))
                x = torch.unsqueeze(x, 0)

        return x


class MLP(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_gc_layers=5):
        super(MLP, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.nns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i == num_gc_layers - 1:
                nn = Linear(dim_in, dim_out)
                bn = torch.nn.BatchNorm1d(dim_out)
            else:
                nn = Linear(dim_in, dim_in)
                bn = torch.nn.BatchNorm1d(dim_in)
            self.nns.append(nn)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_gc_layers):
            x = F.relu(self.nns[i](x))
            if i < self.num_gc_layers - 1:  # no bn for last layer
                x = self.bns[i](torch.squeeze(x))
        return x


class LaGraphNetNode(torch.nn.Module):
    def __init__(self, dim_ft=32, dim_hid=32, dim_eb=32, num_en_layers=5, num_de_layers=5, decoder='gnn'):
        super(LaGraphNetNode, self).__init__()

        self.embedding_layer = Sequential(Linear(dim_ft, dim_eb), ReLU())
        self.bn = torch.nn.BatchNorm1d(dim_eb, affine=False)
        self.encoder = Encoder(dim_in=dim_ft, dim_out=dim_hid, num_gc_layers=num_en_layers)
        if decoder == 'gnn':
            self.decoder = Decoder(dim_in=dim_hid, dim_out=dim_ft, num_gc_layers=num_de_layers)
        elif decoder == 'mlp':
            self.decoder = MLP(dim_in=dim_hid, dim_out=dim_ft, num_gc_layers=num_de_layers)

    def forward(self, x, adj, sparse, args, aug=True):
        x_embed = x
        x_embed_masked, mask = self.mask_node(x_embed, args, aug)

        encoded_x, concat_enc = self.encoder(x_embed_masked, adj, sparse)
        decoded_x = self.decoder(encoded_x, adj, sparse)

        return x_embed, encoded_x, decoded_x, mask

    def mask_node(self, x, args, aug=True):
        if aug is False:
            mask = None

        else:
            _, node_num, feat_dim = x.size()
            if args.mmode == 'partial':
                mask = torch.zeros(node_num, feat_dim)[np.newaxis]
                for i in range(node_num):
                    for j in range(feat_dim):
                        if random.random() < args.mratio:
                            x[0][i][j] = torch.tensor(np.random.normal(loc=0, scale=args.mstd), dtype=torch.float32)
                            mask[0][i][j] = 1

            elif args.mmode == 'whole':
                mask_num = int(node_num * args.mratio)
                idx_mask = np.random.choice(node_num, mask_num, replace=False)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                x[:, idx_mask] = torch.tensor(np.random.normal(loc=0, scale=args.mstd, size=(mask_num, feat_dim)), dtype=torch.float32).to(device)
                mask = torch.zeros(node_num)[np.newaxis]
                mask[:, idx_mask] = 1

        return x, mask

    def get_node_rep(self, x, adj, sparse):
        _, xs = self.encoder(x, adj, sparse)
        rep = torch.cat(xs, 1)

        return rep

    def embed(self, x, adj, sparse):
        x_embed = x
        encoded_rep, concat_enc = self.encoder(x_embed, adj, sparse)

        return encoded_rep.detach()



