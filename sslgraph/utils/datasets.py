import os
import os.path as osp
import shutil
import re

import torch
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.data import Batch, Data
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

from itertools import repeat, product
import numpy as np

from copy import deepcopy

from .feat_expansion import FeatureExpander, CatDegOnehot, get_max_deg


def get_dataset(name, task, sparse=True, feat_str="deg+ak3+reall", root=None):
    if task == "semisupervised":

        if name in ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        if name in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')

        degree = feat_str.find("deg") >= 0
        onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
        onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None

        pre_transform = FeatureExpander(degree=degree, onehot_maxdeg=onehot_maxdeg, AK=0).transform

        dataset = TUDatasetExt("./semi_dataset/dataset", name, task, pre_transform=pre_transform, use_node_attr=True,
                               processed_filename="data_%s.pt" % feat_str)

        dataset_pretrain = TUDatasetExt("./semi_dataset/pretrain_dataset/", name, task, pre_transform=pre_transform,
                                        use_node_attr=True,
                                        processed_filename="data_%s.pt" % feat_str)

        dataset.data.edge_attr = None
        dataset_pretrain.data.edge_attr = None

        return dataset, dataset_pretrain

    elif task == "unsupervised":
        dataset = TUDatasetExt("./unsuper_dataset/", name=name, task=task)
        if feat_str.find("deg") >= 0:
            max_degree = get_max_deg(dataset)
            dataset = TUDatasetExt("./unsuper_dataset/", name=name, task=task,
                                   transform=CatDegOnehot(max_degree), use_node_attr=True)
        return dataset

    else:
        ValueError("Wrong task name")


def get_node_dataset(name, root='./node_dataset/', sparse=True):
    
    full_dataset = Planetoid(root, name)
    train_mask = full_dataset[0].train_mask
    val_mask = full_dataset[0].val_mask
    test_mask = full_dataset[0].test_mask
    return full_dataset, train_mask, val_mask, test_mask


class TUDatasetExt(InMemoryDataset):
    '''
    Used in GraphCL for feature expansion
    '''
    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
#     url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets')

    def __init__(self,
                 root,
                 name,
                 task,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 use_edge_attr=False,
                 cleaned=False,
                 processed_filename='data.pt'
                 ):
        self.processed_filename = processed_filename
        self.name = name
        self.cleaned = cleaned
        self.task = task
        super(TUDatasetExt, self).__init__(root, transform, pre_transform, pre_filter)

        if self.task == "semisupervised":
            self.data, self.slices = torch.load(self.processed_paths[0])
            if self.data.x is not None and not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]
            if self.data.edge_attr is not None and not use_edge_attr:
                num_edge_attributes = self.num_edge_attributes
                self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

        elif self.task == "unsupervised":
            self.data, self.slices = torch.load(self.processed_paths[0])
            if self.data.x is not None and not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]
            if self.data.edge_attr is not None and not use_edge_attr:
                num_edge_attributes = self.num_edge_attributes
                self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            if self.data.x is None:
                edge_index = self.data.edge_index[0, :].numpy()
                _, num_edge = self.data.edge_index.size()
                nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
                nlist.append(edge_index[-1] + 1)

                num_node = np.array(nlist).sum()
                self.data.x = torch.ones((num_node, 1))

                edge_slice = [0]
                k = 0
                for n in nlist:
                    k = k + n
                    edge_slice.append(k)
                self.slices['x'] = torch.tensor(edge_slice)
        else:
            ValueError("Wrong task name")

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return self.processed_filename

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[0], slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.task == "unsupervised":
            node_num = data.edge_index.max()
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        return data


# class NodeDataset(InMemoryDataset):
#     def __init__(self,
#                  root,
#                  name,
#                  sparse=True,
#                  mode='pretrain'):

#         super(NodeDataset, self).__init__()

#         self.root = root
#         self.name = name
#         self.mode = mode
#         self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = self.load()
#         self.features, _ = self.preprocess_features(self.features)
#         self.features = torch.FloatTensor(self.features)
#         self.nb_classes = self.labels.shape[1]

#         if sparse:
#             row_idx, col_idx = self.adj.nonzero()
#             self.edge_index = np.concatenate((np.expand_dims(col_idx, axis=0), np.expand_dims(row_idx, axis=0)), axis=0)
#             self.edge_index = torch.tensor(self.edge_index).long()
#         else:
#             self.adj = self.adj.todense()
#             self.adj = torch.FloatTensor(self.adj[np.newaxis])

#         self.labels = torch.FloatTensor(self.labels[np.newaxis])
#         self.idx_train = torch.LongTensor(self.idx_train)
#         self.idx_val = torch.LongTensor(self.idx_val)
#         self.idx_test = torch.LongTensor(self.idx_test)

#         self.train_lbls = torch.argmax(self.labels[0, self.idx_train], dim=1)
#         self.val_lbls = torch.argmax(self.labels[0, self.idx_val], dim=1)
#         self.test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1)

#     def len(self):
#         return 1

#     def get(self, idx):

#         if self.mode == 'pretrain':
#             data = Batch.from_data_list([Data(x=self.features, edge_index=self.edge_index)])
#             return data, self.nb_classes
#         elif self.mode == 'train':
#             return self.idx_train, self.train_lbls
#         elif self.mode == 'valid':
#             return self.idx_val, self.val_lbls
#         else:
#             return self.idx_test, self.test_lbls

#     def load(self):
#         """Load data."""
#         names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#         objects = []
#         for i in range(len(names)):
#             with open("{}ind.{}.{}".format(self.root, self.name, names[i]), 'rb') as f:
#                 if sys.version_info > (3, 0):
#                     objects.append(pkl.load(f, encoding='latin1'))
#                 else:
#                     objects.append(pkl.load(f))

#         x, y, tx, ty, allx, ally, graph = tuple(objects)
#         test_idx_reorder = self.parse_index_file("{}ind.{}.test.index".format(self.root, self.name))
#         test_idx_range = np.sort(test_idx_reorder)

#         if self.name == 'citeseer':
#             # Fix citeseer dataset (there are some isolated nodes in the graph)
#             # Find isolated nodes, add them as zero-vecs into the right position
#             test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
#             tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#             tx_extended[test_idx_range - min(test_idx_range), :] = tx
#             tx = tx_extended
#             ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#             ty_extended[test_idx_range - min(test_idx_range), :] = ty
#             ty = ty_extended

#         features = sp.vstack((allx, tx)).tolil()
#         features[test_idx_reorder, :] = features[test_idx_range, :]
#         adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#         labels = np.vstack((ally, ty))
#         labels[test_idx_reorder, :] = labels[test_idx_range, :]

#         idx_test = test_idx_range.tolist()
#         idx_train = range(len(y))
#         idx_val = range(len(y), len(y) + 500)

#         return adj, features, labels, idx_train, idx_val, idx_test

#     def parse_index_file(self, filename):
#         """Parse index file."""
#         index = []
#         for line in open(filename):
#             index.append(int(line.strip()))
#         return index

#     def sparse_to_tuple(self, sparse_mx, insert_batch=False):
#         """Convert sparse matrix to tuple representation."""
#         """Set insert_batch=True if you want to insert a batch dimension."""

#         def to_tuple(mx):
#             if not sp.isspmatrix_coo(mx):
#                 mx = mx.tocoo()
#             if insert_batch:
#                 coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
#                 values = mx.data
#                 shape = (1,) + mx.shape
#             else:
#                 coords = np.vstack((mx.row, mx.col)).transpose()
#                 values = mx.data
#                 shape = mx.shape
#             return coords, values, shape

#         if isinstance(sparse_mx, list):
#             for i in range(len(sparse_mx)):
#                 sparse_mx[i] = to_tuple(sparse_mx[i])
#         else:
#             sparse_mx = to_tuple(sparse_mx)

#         return sparse_mx

#     def preprocess_features(self, features):
#         """Row-normalize feature matrix and convert to tuple representation"""
#         rowsum = np.array(features.sum(1))
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         features = r_mat_inv.dot(features)
#         return features.todense(), self.sparse_to_tuple(features)
