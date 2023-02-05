# Author: Youzhi Luo (yzluo@tamu.edu)

import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.utils import degree

class DegreeTrans(object):
    def __init__(self, dataset, in_degree=False):
        self.max_degree = None
        self.mean = None
        self.std = None
        self.in_degree = in_degree
        self._statistic(dataset)
    
    def _statistic(self, dataset):
        degs = []
        max_degree = 0
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        
        self.max_degree = max_degree
        deg = torch.cat(degs, dim=0).to(torch.float)
        self.mean, self.std = deg.mean().item(), deg.std().item()
    
    def __call__(self, data):
        if data.x is not None:
            return data
        if self.max_degree < 1000:
            idx = data.edge_index[1 if self.in_degree else 0]
            deg = torch.clamp(degree(idx, data.num_nodes, dtype=torch.long), min=0, max=self.max_degree)
            deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)
            data.x = deg
        else:
            deg = degree(data.edge_index[0], dtype=torch.float)
            deg = (deg - self.mean) / self.std
            data.x = deg.view(-1, 1)
        
        return data        


class AUG_trans(object):
    def __init__(self, augmenter, device, pre_trans=None, post_trans=None):
        self.augmenter = augmenter
        self.pre_trans = pre_trans
        self.post_trans = post_trans
        self.device = device
    
    def __call__(self, data):
        # data = data.to(self.device)
        if self.pre_trans:
            data = self.pre_trans(data)
        new_data = self.augmenter(data)[0]
        if self.post_trans:
            new_data = self.post_trans(new_data)
        return new_data


class Subset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        data = self.subset[index]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def __len__(self):
        return len(self.subset)


class DoubleSet(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.data_len = len(self.dataset)
    
    def __getitem__(self, index):
        anchor_data = self.dataset[index]

        pos_index = random.sample(range(self.data_len), 1)[0]
        while pos_index == index:
            pos_index = random.sample(range(self.data_len), 1)[0]
        pos_data = self.dataset[pos_index]
        
        if self.transform is not None:
            anchor_data, pos_data = self.transform(anchor_data), self.transform(pos_data)
        
        lambd = torch.tensor([np.random.beta(1.0, 1.0)])
        return anchor_data, pos_data, lambd
    
    def __len__(self):
        return self.data_len


class TripleSet(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self._preprocess()
    
    def _preprocess(self):
        self.label_to_index_list = {}
        for i, data in enumerate(self.dataset):
            y = int(data.y.item())
            if not y in self.label_to_index_list:
                self.label_to_index_list[y] = [i]
            else:
                self.label_to_index_list[y].append(i)

    def __getitem__(self, index):
        anchor_data = self.dataset[index]
        anchor_label = int(anchor_data.y.item())

        pos_index = random.sample(self.label_to_index_list[anchor_label], 1)[0]
        while pos_index == index:
            pos_index = random.sample(self.label_to_index_list[anchor_label], 1)[0]
        
        neg_label = random.sample(self.label_to_index_list.keys(), 1)[0]
        while neg_label == anchor_label:
            neg_label = random.sample(self.label_to_index_list.keys(), 1)[0]
        neg_index = random.sample(self.label_to_index_list[neg_label], 1)[0]

        pos_data, neg_data = self.dataset[pos_index], self.dataset[neg_index]

        if self.transform is not None:
            anchor_data, pos_data, neg_data = self.transform(anchor_data), self.transform(pos_data), self.transform(neg_data)
        
        return anchor_data, pos_data, neg_data
    
    def __len__(self):
        return len(self.dataset)