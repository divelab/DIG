import csv
import numpy as np
from rdkit import Chem
import os, shutil, re, torch, json, ast
import os.path as osp
import pandas as pd
# import scipy.sparse as sp
from torch.utils.data import Dataset
import networkx as nx
from dig.ggraph.dataset import PygDataset

from itertools import repeat, product
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import to_networkx

bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
zinc_atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
qm9_atom_list = [6, 7, 8, 9]

class QM9(PygDataset):
    
    r"""An `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset.
    
    Args:
        root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
        prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`penalized_logp`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
    """
        
    def __init__(self,
                 root='./',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        name='qm9_property'
        super(QM9, self).__init__(root, name, prop_name, conf_dict, 
                                  transform, pre_transform, pre_filter, 
                                  processed_filename, use_aug, one_shot)
        
        
class ZINC250k(PygDataset):
    
    r"""An `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`ZINC250k` dataset.
    
    Args:
        root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
        prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`penalized_logp`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
    """
    
    def __init__(self,
                 root='./',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        name='zinc250k_property'
        super(ZINC250k, self).__init__(root, name, prop_name, conf_dict, 
                                  transform, pre_transform, pre_filter, 
                                  processed_filename, use_aug, one_shot)
        

class ZINC800(PygDataset):
    
    r"""An `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`ZINC800` dataset.
    
    Args:
        root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
        method (string, optional): Method name for :obj:`ZINC800` dataset, can be either :obj:`jt` or :obj:`graphaf`. (default: :obj:`jt`)
        prop_name (string, optional): The molecular property desired and used as the optimization target.(default: :obj:`penalized_logp`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
    """
    
    def __init__(self,
                 root='./',
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
        
        name='zinc800'
        
        if method in ['jt', 'graphaf']:
            name = 'zinc_800' + '_' + method
        else:
            error_mssg = 'Invalid method type {}.\n'.format(method)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(['jt', 'graphaf'])
            raise ValueError(error_mssg)
        super(ZINC800, self).__init__(root, name, prop_name, conf_dict, 
                                  transform, pre_transform, pre_filter, 
                                  processed_filename, use_aug, one_shot)
        
class MOSES(PygDataset):
    
    r"""An `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`MOSES` dataset.
    
    Args:
        root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
        prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
    """
    
    def __init__(self,
                 root='./',
                 prop_name=None,
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        
        name='moses'
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
    
   