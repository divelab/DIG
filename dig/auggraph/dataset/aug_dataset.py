# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.utils import degree


class DegreeTrans(object):
    r"""
    This class is used to add vertex degree based node features to graphs.
    This is usually used to preprocess the graph datasets that do not have
    node features.
    """
    def __init__(self, dataset, in_degree=False):
        self.max_degree = None
        self.mean = None
        self.std = None
        self.in_degree = in_degree
        self._statistic(dataset)
    
    def _statistic(self, dataset):
        r"""
        This function computes statistics over all nodes in all sample graphs.
        These statistics are maximum, mean, and standard deviation.

        Args:
            dataset (:class:`torch.utils.data.Dataset`): The dataset containing
            all sample graphs.
        """
        degs = []
        max_degree = 0
        for data in dataset:
            print(type(data))
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        
        self.max_degree = max_degree
        deg = torch.cat(degs, dim=0).to(torch.float)
        self.mean, self.std = deg.mean().item(), deg.std().item()
    
    def __call__(self, data):
        r"""
            This is the main function that adds vertex degree based node
            features to the given graph.

            Args:
                data (:class:`torch_geometric.data.data.Data`): The graph
                with vertex degrees as node features.
        """
        if data.x is not None:
            return data
        if self.max_degree < 1000:
            idx = data.edge_index[1 if self.in_degree else 0]
            deg = torch.clamp(degree(idx, data.num_nodes, dtype=torch.long),
                              min=0, max=self.max_degree)
            deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)
            data.x = deg
        else:
            deg = degree(data.edge_index[0], dtype=torch.float)
            deg = (deg - self.mean) / self.std
            data.x = deg.view(-1, 1)
        return data


class AUG_trans(object):
    r"""
    This class generates an augmentation from a given sample.

    Args:
        augmenter (function): This method generates an augmentation from the
            given sample.
        device (str): The device on which the data will be processed.
        pre_trans (function, optional): This transformation is applied on the
            original sample before an augmentation is generated. Default is
            None.
        post_trans (function, optional): This transformation is applied on the
            generated augmented sample. Default is None.
    """
    def __init__(self, augmenter, device, pre_trans=None, post_trans=None):
        self.augmenter = augmenter
        self.pre_trans = pre_trans
        self.post_trans = post_trans
        self.device = device

    def __call__(self, data):
        r"""
        This is the main function that generates an augmentation from a given
        sample.

        Args:
            data: The given data sample.
        Returns:
            A transformed graph.
        """
        if self.pre_trans:
            data = self.pre_trans(data)
        new_data = self.augmenter(data)[0]
        if self.post_trans:
            new_data = self.post_trans(new_data)
        return new_data


class Subset(Dataset):
    r"""
    This class is used to create of a subset of a dataset.

    Args:
        subset (:class:`torch.utils.data.Dataset`): The given dataset subset.
        transform (function, optional): A transformation applied on each
            sample of the dataset before it will be used. Default is None.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        r"""
        This method returns the sample at the given index in the subset.

        Args:
            index (int): The index in the subset of the required sample.
        """
        data = self.subset[index]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def __len__(self):
        r"""
        Returns the number of samples in the subset.
        """
        return len(self.subset)

class TripleSet(Dataset):
    r"""
    This class inherits from the :class:`torch.utils.data.Dataset` class and in
    addition to each anchor sample, it returns a random positive and negative
    sample from the dataset. A positive sample has the same label as the
    anchor sample and a negative sample has a different label than the anchor
    sample.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset for which the
            triple set will be created.
        transform (function, optional): A transformation that is applied on all
            original samples. In other words, this transformation is applied
            to the anchor, positive, and negative sample. Default is None.
    """
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
        r"""
        For a given index, this sample returns the original/anchor sample from
        the dataset at that index and a corresponding positive, and negative
        sample.

        Args:
            index (int): The index of the anchor sample in the dataset.

        Returns:
            A tuple consisting of the anchor sample, a positive
            sample, and a negative sample respectively.
        """
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
            anchor_data, pos_data, neg_data = self.transform(anchor_data), \
                self.transform(pos_data), self.transform(neg_data)
        
        return anchor_data, pos_data, neg_data

    def __len__(self):
        r"""
            Returns:
                 The number of samples in the original dataset.
        """
        return len(self.dataset)