import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool

import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys


class Encoder(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_gc_layers=5):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i == 0:
                nn = Sequential(Linear(dim_in, dim_out), ReLU(), Linear(dim_out, dim_out))
            else:
                nn = Sequential(Linear(dim_out, dim_out), ReLU(), Linear(dim_out, dim_out))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim_out)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        # last x is the encoded node reps for all nodes in the batch
        return x, xs


class Decoder(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_gc_layers=5):
        super(Decoder, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i == num_gc_layers - 1:
                nn = Sequential(Linear(dim_in, dim_in), ReLU(), Linear(dim_in, dim_out))
                bn = torch.nn.BatchNorm1d(dim_out)
            else:
                nn = Sequential(Linear(dim_in, dim_in), ReLU(), Linear(dim_in, dim_in))
                bn = torch.nn.BatchNorm1d(dim_in)
            conv = GINConv(nn)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            if i < self.num_gc_layers - 1:  # no bn for last layer
                x = self.bns[i](x)
            # x = self.bns[i](x)

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
                x = self.bns[i](x)
        return x


class LaGraphNet(torch.nn.Module):
    def __init__(self, dim=32, num_en_layers=5, num_de_layers=5, pool='add', decoder='gnn'):
        super(LaGraphNet, self).__init__()

        self.pool = pool
        self.encoder = Encoder(dim_in=dim, dim_out=32, num_gc_layers=num_en_layers)
        if decoder == 'gnn':
            self.decoder = Decoder(dim_in=32, dim_out=dim, num_gc_layers=num_de_layers)
        elif decoder == 'mlp':
            self.decoder = MLP(dim_in=32, dim_out=dim, num_gc_layers=num_de_layers)

    def forward(self, x, edge_index, batch):
        encoded_rep, xs = self.encoder(x, edge_index, batch)
        decoded_rep = self.decoder(encoded_rep, edge_index, batch)

        if self.pool == 'add':
            xpool = [global_add_pool(x, batch) for x in xs]
        elif self.pool == 'mean':
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return encoded_rep, decoded_rep, global_rep

    def get_global_rep(self, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        global_reps = []
        y = []
        with torch.no_grad():
            for data in loader:
                # data = data[0]  # just need original graph
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)

                _, xs = self.encoder(x, edge_index, batch)
                if self.pool == 'add':
                    xpool = [global_add_pool(x, batch) for x in xs]
                elif self.pool == 'mean':
                    xpool = [global_mean_pool(x, batch) for x in xs]
                global_rep = torch.cat(xpool, 1)

                global_reps.append(global_rep.cpu().numpy())
                y.append(data.y.cpu().numpy())
        global_reps = np.concatenate(global_reps, 0)
        y = np.concatenate(y, 0)
        return global_reps, y

    def save_network(self, save_pth):
        torch.save(self.state_dict(), save_pth)

    def load_network(self, load_pth):
        self.load_state_dict(torch.load(load_pth))




