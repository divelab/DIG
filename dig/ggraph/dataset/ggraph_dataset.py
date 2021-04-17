import csv
import numpy as np
from rdkit import Chem
import os, shutil, re, torch, json, ast
import os.path as osp
import pandas as pd
# import scipy.sparse as sp
from torch.utils.data import Dataset
import networkx as nx
from PygDataset import PygDataset

from itertools import repeat, product
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import to_networkx

bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
zinc_atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
qm9_atom_list = [6, 7, 8, 9]

class QM9(PygDataset):
    def __init__(self,
                 root='./',
                 name='qm9_property',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        """
        Pytorch Geometric data interface for molecule datasets.
        param root: root directory where the dataset should be saved.
        param name: the name of the dataset you want to use.
        param prop_name: the molecular property desired and used as the optimization target.
        param conf_dict: dictionary that stores all the configuration for the corresponding dataset. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        param use_aug: whether data augmentation is used, default is False
        param one_shot: 
                   
        All the rest of parameters of PygDataset follows the use in 'InMemoryDataset' from torch_geometric.data.
        Documentation can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html.
        """

        super(QM9, self).__init__(root, name, prop_name, conf_dict, 
                                  transform, pre_transform, pre_filter, 
                                  processed_filename, use_aug, one_shot)
        
        
class ZINC250k(PygDataset):
    def __init__(self,
                 root='./',
                 name='zinc250k_property',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        """
        Pytorch Geometric data interface for molecule datasets.
        param root: root directory where the dataset should be saved.
        param name: the name of the dataset you want to use.
        param prop_name: the molecular property desired and used as the optimization target.
        param conf_dict: dictionary that stores all the configuration for the corresponding dataset. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        param use_aug: whether data augmentation is used, default is False
        param one_shot: 
                   
        All the rest of parameters of PygDataset follows the use in 'InMemoryDataset' from torch_geometric.data.
        Documentation can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html.
        """

        super(ZINC250k, self).__init__(root, name, prop_name, conf_dict, 
                                  transform, pre_transform, pre_filter, 
                                  processed_filename, use_aug, one_shot)
        

class ZINC800(PygDataset):
    def __init__(self,
                 root='./',
                 name='zinc800',
                 method='jt',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        """
        Pytorch Geometric data interface for molecule datasets.
        param root: root directory where the dataset should be saved.
        param name: the name of the dataset you want to use.
        param prop_name: the molecular property desired and used as the optimization target.
        param conf_dict: dictionary that stores all the configuration for the corresponding dataset. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        param use_aug: whether data augmentation is used, default is False
        param one_shot: 
                   
        All the rest of parameters of PygDataset follows the use in 'InMemoryDataset' from torch_geometric.data.
        Documentation can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html.
        """
        if method in ['jt', 'graphaf']:
            name = 'zinc_800' + '_' + method
        else:
            error_mssg = 'Invalid method name {}.\n'.format(method)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(['jt', 'graphaf'])
            raise ValueError(error_mssg)
        super(ZINC800, self).__init__(root, name, prop_name, conf_dict, 
                                  transform, pre_transform, pre_filter, 
                                  processed_filename, use_aug, one_shot)
        
class MOSES(PygDataset):
    def __init__(self,
                 root='./',
                 name='moses',
                 prop_name=None,
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        """
        Pytorch Geometric data interface for molecule datasets.
        param root: root directory where the dataset should be saved.
        param name: the name of the dataset you want to use.
        param prop_name: the molecular property desired and used as the optimization target.
        param conf_dict: dictionary that stores all the configuration for the corresponding dataset. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        param use_aug: whether data augmentation is used, default is False
        param one_shot: 
                   
        All the rest of parameters of PygDataset follows the use in 'InMemoryDataset' from torch_geometric.data.
        Documentation can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html.
        """

        super(MOSES, self).__init__(root, name, prop_name, conf_dict,transform, pre_transform, pre_filter, 
                                  processed_filename, use_aug, one_shot)
                        
if __name__ == '__main__':
    test = QM9()
    print(test[0])
    import pdb; pdb.set_trace()
    
    test = ZINC250k()
    print(test[0])
    
    test = ZINC800(method='jt')
    print(test[0])
    
    test = MOSES()
    print(test[0])
    
   