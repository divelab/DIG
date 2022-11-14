from typing import Tuple

import torch
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import GNNBenchmarkDataset, Yelp, Flickr, Reddit2
from ogb.nodeproppred import PygNodePropPredDataset


def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_yelp(root: str) -> Tuple[Data, int, int]:
    dataset = Yelp(f'{root}/YELP', pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr', pre_transform=T.ToSparseTensor())
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2', pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_sbm(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train',
                                  pre_transform=T.ToSparseTensor())
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes


def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name)
    elif name.lower() == 'reddit':
        return get_reddit(root)
    elif name.lower() == 'flickr':
        return get_flickr(root)
    elif name.lower() == 'yelp':
        return get_yelp(root)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root)
    elif name.lower() in ['ogbn-products', 'products']:
        return get_products(root)
    else:
        raise NotImplementedError
