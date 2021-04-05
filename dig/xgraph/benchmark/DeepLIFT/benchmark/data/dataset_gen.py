"""
FileName: dataset_gen.py
Description: dataset generator
Time: 2020/12/28 19:16
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from definitions import ROOT_DIR
import os
import pickle as pkl
import numpy as np


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_node_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data



class BA_LRP(InMemoryDataset):

    def __init__(self, root, num_per_class, transform=None, pre_transform=None):
        self.num_per_class = num_per_class
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data{self.num_per_class}.pt']

    def gen_class1(self):
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[0]], dtype=torch.float))

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = prob_dist.sample().squeeze()
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)

        return data

    def gen_class2(self):
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[1]], dtype=torch.float))
        epsilon = 1e-30

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg_reciprocal = torch.stack([1 / ((data.edge_index[0] == node_idx).float().sum() + epsilon) for node_idx in range(i)], dim=0)
            sum_deg_reciprocal = deg_reciprocal.sum(dim=0, keepdim=True)
            probs = (deg_reciprocal / sum_deg_reciprocal).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = -1
            for _ in range(1 if i % 5 != 4 else 2):
                new_node_pick = prob_dist.sample().squeeze()
                while new_node_pick == node_pick:
                    new_node_pick = prob_dist.sample().squeeze()
                node_pick = new_node_pick
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)

        return data

    def process(self):
        data_list = []
        for i in range(self.num_per_class):
            data_list.append(self.gen_class1())
            data_list.append(self.gen_class2())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BA_Shape(InMemoryDataset):

    def __init__(self, root, num_base_node, num_shape, transform=None, pre_transform=None):
        self.num_base_node = num_base_node
        self.num_shape = num_shape
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        indices = []
        num_classes = 4
        train_percent = 0.7
        for i in range(num_classes):
            index = (self.data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)

        rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        self.data.train_mask = index_to_mask(train_index, size=self.data.num_nodes)
        self.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=self.data.num_nodes)
        self.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=self.data.num_nodes)

        self.data, self.slices = self.collate([self.data])

    @property
    def processed_file_names(self):
        return [f'data{"_debug2"}.pt']

    def gen(self):
        x = torch.tensor([[1], [1], [1], [1], [1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[5, 5, 5, 5, 5, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        # --- generate basic BA graph ---
        for i in range(6, self.num_base_node):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_picks = []
            for _ in range(5):
                node_pick = prob_dist.sample().squeeze()
                while node_pick in node_picks:
                    node_pick = prob_dist.sample().squeeze()
                node_picks.append(node_pick)
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)

        data.y = torch.zeros(data.x.shape[0], dtype=torch.long)

        # --- add shapes ---
        house_x = torch.tensor([[1] for _ in range(5)], dtype=torch.float)
        house_y = torch.tensor([1, 2, 2, 3, 3], dtype=torch.long)
        house_edge_index = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4],
                                         [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3]], dtype=torch.long)
        house_data = Data(x=house_x, edge_index=house_edge_index, y = house_y)
        house_connect_probs = torch.tensor([[0.2 for _ in range(5)]])
        house_connect_dist = torch.distributions.Categorical(house_connect_probs)
        base_connect_probs = torch.tensor([[1.0 / self.num_base_node]]).repeat(1, self.num_base_node)
        base_connect_dist = torch.distributions.Categorical(base_connect_probs)
        for i in range(self.num_shape):
            data.edge_index = torch.cat([data.edge_index, house_data.edge_index + data.x.shape[0]], dim=1)
            house_pick = house_connect_dist.sample().squeeze() + data.x.shape[0]
            base_pick = base_connect_dist.sample().squeeze()
            data.x = torch.cat([data.x, house_data.x], dim=0)
            data.y = torch.cat([data.y, house_data.y], dim=0)
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[base_pick, house_pick], [house_pick, base_pick]], dtype=torch.long)], dim=1)

        # --- add random edges ---
        probs = torch.tensor([[1.0 / data.x.shape[0]]]).repeat(2, data.x.shape[0])
        dist = torch.distributions.Categorical(probs)
        for i in range(data.x.shape[0] // 10):
            node_pair = dist.sample().squeeze()
            if node_pair[0] != node_pair[1] and \
                    (data.edge_index[1][data.edge_index[0] == node_pair[0]] == node_pair[1]).int().sum() == 0:
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pair[0], node_pair[1]], [node_pair[1], node_pair[0]]],
                                                          dtype=torch.long)], dim=1)

        return data

    def process(self):
        data_list = []
        data_list.append(self.gen())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
