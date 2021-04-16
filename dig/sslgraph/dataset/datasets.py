import os, re, torch
import os.path as osp
import numpy as np

from itertools import repeat, product
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data import Batch, Data

from .TUDataset import TUDatasetExt
from .feat_expansion import FeatureExpander, CatDegOnehot, get_max_deg


def get_dataset(name, task, feat_str="deg", root=None):
    r"""A pre-implemented function to retrieve graph datasets from TUDataset.
    Depending on evaluation tasks, different node feature augmentation will
    be applied following `GraphCL <https://arxiv.org/abs/2010.13902>`_.

    Args:
        name (string): The `name <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the dataset.
        task (string): The evaluation task. Either 'semisupervised' or
            `unsupervised`.
        feat_str (bool, optional): The node feature augmentations to be applied,
            *e.g.*, degrees and centrality. (default: :obj:`deg`)
        root (string, optional): Root directory where the dataset should be saved.
            (default: :obj:`None`)
        
    :rtype: :class:`Dataset` if task is unsupervised, or (:class:`Dataset`, 
        :class:`Dataset`) if task is semisupervised.
    """

    root = "." if root is None else root
    if task == "semisupervised":

        if name in ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        if name in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')

        degree = feat_str.find("deg") >= 0
        onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
        onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None

        pre_transform = FeatureExpander(degree=degree, 
                                        onehot_maxdeg=onehot_maxdeg, AK=0).transform

        dataset = TUDatasetExt(root+"/semi_dataset/dataset", name, task, 
                               pre_transform=pre_transform, use_node_attr=True,
                               processed_filename="data_%s.pt" % feat_str)

        dataset_pretrain = TUDatasetExt(root+"/semi_dataset/pretrain_dataset/", name, task, 
                                        pre_transform=pre_transform, use_node_attr=True,
                                        processed_filename="data_%s.pt" % feat_str)

        dataset.data.edge_attr = None
        dataset_pretrain.data.edge_attr = None

        return dataset, dataset_pretrain

    elif task == "unsupervised":
        dataset = TUDatasetExt(root+"/unsuper_dataset/", name=name, task=task)
        if feat_str.find("deg") >= 0:
            max_degree = get_max_deg(dataset)
            dataset = TUDatasetExt(root+"./unsuper_dataset/", name=name, task=task,
                                   transform=CatDegOnehot(max_degree), use_node_attr=True)
        return dataset

    else:
        ValueError("Wrong task name")


def get_node_dataset(name, root='./node_dataset/'):
    r"""A pre-implemented function to retrieve node datasets from Planetoid. It
    returns the Dataset object and the masks for training, val, and test.

    Args:
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        root (string, optional): Root directory where the dataset should be saved.
            (default: :obj:`None`)
        
    :rtype: (:class:`Dataset`, :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
    """
    
    full_dataset = Planetoid(root, name)
    train_mask = full_dataset[0].train_mask
    val_mask = full_dataset[0].val_mask
    test_mask = full_dataset[0].test_mask
    return full_dataset, train_mask, val_mask, test_mask

