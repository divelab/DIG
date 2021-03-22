"""
FileName: dataset.py
Description: Dataset utils
Time: 2020/7/28 11:48
Project: GNN_benchmark
Author: Shurui Gui
"""

from torch_geometric.datasets import MoleculeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Dataset
import torch
from torch.utils.data import random_split
from benchmark import data_args
from definitions import ROOT_DIR
from benchmark.kernel.utils import Metric
from benchmark.data.dataset_gen import BA_LRP, BA_Shape
import os, sys
import copy



def load_dataset(name: str) -> dir:
    """
    Load dataset.
    :param name: dataset's name. Possible options:("ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV",
    "BACE", "BBPB", "Tox21", "ToxCast", "SIDER", "ClinTox")
    :return: torch_geometric.dataset object
    """
    molecule_set = ["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV",
                    "BACE", "BBPB", "Tox21", "ToxCast", "SIDER", "ClinTox"]
    molecule_set = [x.lower() for x in molecule_set]
    name = name.lower()

    # set Metrics: loss and score based on dataset's name
    Metric.set_loss_func(name)
    Metric.set_score_func(name)


    # To Do: use transform to argument data
    if name in molecule_set:
        data_args.dataset_type = 'mol'
        data_args.model_level = 'graph'

        dataset = MoleculeNet(root=os.path.abspath(os.path.join(ROOT_DIR, '..', 'datasets')), name=name)
        dataset.data.x = dataset.data.x.to(torch.float32)
        data_args.dim_node = dataset.num_node_features
        data_args.dim_edge = dataset.num_edge_features
        data_args.num_targets = dataset.num_classes  # This so-called classes are actually targets.

        # Define models' output shape.
        if Metric.cur_task == 'bcs':
            data_args.num_classes = 2
        elif Metric.cur_task == 'reg':
            data_args.num_classes = 1

        assert data_args.target_idx != -1, 'Explaining on multi tasks is meaningless.'
        assert data_args.target_idx <= dataset.data.y.shape[1], 'No such target in the dataset.'

        dataset.data.y = dataset.data.y[:, data_args.target_idx]
        data_args.num_targets = 1

        dataset_len = len(dataset)
        dataset_split = [int(dataset_len * data_args.dataset_split[0]),
                         int(dataset_len * data_args.dataset_split[1]),
                         0]
        dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
        train_set, val_set, test_set = \
            random_split(dataset, dataset_split)

        return {'train': train_set, 'val': val_set, 'test': test_set}

    elif name == 'ba_lrp':
        data_args.dataset_type = 'syn'
        data_args.model_level = 'graph'

        dataset = BA_LRP(root=os.path.join(ROOT_DIR, '..', 'datasets', 'ba_lrp'),
                         num_per_class=10000)
        dataset.data.x = dataset.data.x.to(torch.float32)
        data_args.dim_node = dataset.num_node_features
        data_args.dim_edge = dataset.num_edge_features
        data_args.num_targets = dataset.num_classes  # This so-called classes are actually targets.

        # Define models' output shape.
        if Metric.cur_task == 'bcs':
            data_args.num_classes = 2
        elif Metric.cur_task == 'reg':
            data_args.num_classes = 1

        assert data_args.target_idx != -1, 'Explaining on multi tasks is meaningless.'
        assert data_args.target_idx <= dataset.data.y.shape[1], 'No such target in the dataset.'

        dataset.data.y = dataset.data.y[:, data_args.target_idx]
        data_args.num_targets = 1

        dataset_len = len(dataset)
        dataset_split = [int(dataset_len * data_args.dataset_split[0]),
                         int(dataset_len * data_args.dataset_split[1]),
                         0]
        dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
        train_set, val_set, test_set = \
            random_split(dataset, dataset_split)

        return {'train': train_set, 'val': val_set, 'test': test_set}
    elif name == 'ba_shape':
        data_args.dataset_type = 'syn'
        data_args.model_level = 'node'

        dataset = BA_Shape(root=os.path.join(ROOT_DIR, '..', 'datasets', 'ba_shape'),
                         num_base_node=300, num_shape=80)
        dataset.data.x = dataset.data.x.to(torch.float32)
        data_args.dim_node = dataset.num_node_features
        data_args.dim_edge = dataset.num_edge_features
        data_args.num_targets = 1

        # Define models' output shape.
        if Metric.cur_task == 'bcs':
            data_args.num_classes = 2
        elif Metric.cur_task == 'reg':
            data_args.num_classes = 1
        else:
            data_args.num_classes = dataset.num_classes

        assert data_args.target_idx != -1, 'Explaining on multi tasks is meaningless.'
        if data_args.model_level != 'node':

            assert data_args.target_idx <= dataset.data.y.shape[1], 'No such target in the dataset.'

            dataset.data.y = dataset.data.y[:, data_args.target_idx]
            data_args.num_targets = 1

            dataset_len = len(dataset)
            dataset_split = [int(dataset_len * data_args.dataset_split[0]),
                             int(dataset_len * data_args.dataset_split[1]),
                             0]
            dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
            train_set, val_set, test_set = \
                random_split(dataset, dataset_split)

            return {'train': train_set, 'val': val_set, 'test': test_set}
        else:
            train_set = dataset
            val_set = copy.deepcopy(dataset)
            test_set = copy.deepcopy(dataset)
            train_set.data.mask = train_set.data.train_mask
            train_set.slices['mask'] = train_set.slices['train_mask']
            val_set.data.mask = val_set.data.val_mask
            val_set.slices['mask'] = val_set.slices['val_mask']
            test_set.data.mask = test_set.data.test_mask
            test_set.slices['mask'] = test_set.slices['test_mask']
            return {'train': train_set, 'val': val_set, 'test': test_set}
    print(f'#E#Dataset {name} does not exist.')
    sys.exit(1)


def create_dataloader(dataset):

    if data_args.model_level == 'node':
        loader = {'train': DataLoader(dataset['train'], batch_size=1, shuffle=True),
                  'val': DataLoader(dataset['val'], batch_size=1, shuffle=True),
                  'test': DataLoader(dataset['test'], batch_size=1, shuffle=False),
                  'explain': DataLoader(dataset['test'], batch_size=1, shuffle=False)}
    else:
        loader = {'train': DataLoader(dataset['train'], batch_size=data_args.train_bs, shuffle=True),
                  'val': DataLoader(dataset['val'], batch_size=data_args.val_bs, shuffle=True),
                  'test': DataLoader(dataset['test'], batch_size=data_args.test_bs, shuffle=False),
                  'explain': DataLoader(dataset['test'], batch_size=data_args.x_bs, shuffle=False)}

    return loader
