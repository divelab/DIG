import csv
import numpy as np
from rdkit import Chem
import os, shutil, re, torch
import os.path as osp
import pandas as pd
# import scipy.sparse as sp
from torch.utils.data import Dataset
import networkx as nx

from itertools import repeat, product
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import to_networkx

def graph_data_obj_to_nx_simple(data, use_aug=False):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    if use_aug:
        local_perm = np.random.permutation(num_atoms)
    else:
        local_perm = np.arange(num_atoms)

    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[local_perm[i]]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = np.where(local_perm == int(edge_index[0, j]))[0][0]
        end_idx = np.where(local_perm == int(edge_index[1, j]))[0][0]
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G, local_perm

class PygDataset(InMemoryDataset):
    def __init__(self,
                 root='../datasets',
                 name='qm9',
                 prop_name='penalized_logp',
                 num_max_node=38,
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False
                 ):
        """
        Pytorch Geometric data interface for molecule datasets.
        param root: root directory where the dataset should be saved.
        param name: the name of the dataset you want to use.
        param prop_name: the molecular property desired and used as the optimization target.
        param num_max_node: the maximum number of nodes (atoms) among all molecules
        param conf_dict: dictionary that stores all the configuration for the corresponding dataset. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        param use_aug: whether data augmentation is used, default is False
                   
        All the rest of parameters of PygDataset follows the use in 'InMemoryDataset' from torch_geometric.data.
        Documentation can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html.
        """
        
        
        self.processed_filename = processed_filename
        self.root = root
        self.name = name
#         self.edge_unroll = edge_unroll    # not used
        self.prop_name = prop_name
        self.num_max_node = num_max_node
        self.use_aug = use_aug
        
        if conf_dict is None:                        
            config_file = pd.read_csv('config.csv', index_col = 0)
            if not self.name in config_file:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(config_file.keys())
                raise ValueError(error_mssg)
            config = config_file[self.name]
            
        else:
            config = conf_dict
            
        self.url = config['url']
        self.available_prop = str(prop_name) in config['prop_list']
        self.smile_col = config['smile']
        
        super(PygDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices, self.all_smiles = torch.load(self.processed_paths[0])
        
    @property
    def num_nodes(self):
        return len(self.data.x)
    
#     @property
#     def num_max_node(self):
#         return (self.slices['x'][1:]-self.slices['x'][:-1]).max().item()
    
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
    def atom_list(self):
        return torch.unique(self.data.x[:,0])
    
    @property
    def raw_dir(self):
#         name = 'raw'
#         return osp.join(self.root, self.name, name)
        return osp.join(self.root)
    
    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
#         names = ['A', 'graph_indicator', 'node_attributes', 'edge_attributes', 'graph_attributes']
#         return ['{}_{}.txt'.format(self.name, name) for name in names]
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return self.processed_filename
    
#     def download(self):
#         url = self.url
#         folder = osp.join(self.root, self.name)
#         path = download_url('{}/{}.zip'.format(url, self.name), folder)
#         extract_zip(path, folder)
#         os.unlink(path)
#         shutil.rmtree(self.raw_dir)
#         os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
#         self.data, self.slices = read_tu_data(self.raw_dir, self.name)
            
        ''' If you want to use your own csv to process, do not use read_tu_data '''
        self.data, self.slices = self.pre_process()
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
            
        torch.save((self.data, self.slices, self.all_smiles), self.processed_paths[0])
        
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
    
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
            
        data['smile'] = self.all_smiles[idx]
            
        # bfs-searching order
        root = 0
        G, local_perm = graph_data_obj_to_nx_simple(data, use_aug=self.use_aug)
        edges = nx.bfs_edges(G, root)
        nodes = [root] + [v for u, v in edges]
        if self.use_aug:
            nodes = local_perm[nodes]
        
        data['bfs_perm_origin'] = torch.Tensor(nodes).long()

        return data
    
    def pre_process(self):
        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=',', dtype='str')
        smile_list = list(input_df[self.smile_col])
        if self.available_prop:
                prop_list = list(input_df[self.prop_name])
                
        self.all_smiles = smile_list
        data_list = []
        
        for i in range(len(smile_list)):
            smile = smile_list[i]
            mol = Chem.MolFromSmiles(smile)
            Chem.Kekulize(mol)
            num_atom = mol.GetNumAtoms()
            if num_atom > self.num_max_node:
                continue
            else:
                # atoms
                num_atom_features = 2   # atom type,  chirality tag
                atom_features_list = []
                for atom in mol.GetAtoms():
                    atom_feature = [atom.GetAtomicNum()] + [atom.GetChiralTag()]
                    atom_features_list.append(atom_feature)
                x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

                # bonds
                num_bond_features = 2   # bond type, bond direction
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    edge_feature = [bond.GetBondType()] + [bond.GetBondDir()]
                    edges_list.append((i, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, i))
                    edge_features_list.append(edge_feature)

                edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
                                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                if self.available_prop:
                    data.y = torch.tensor([float(prop_list[i])])
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices
    
if __name__ == '__main__':
    name = pd.read_csv('config.csv', index_col = 0)
    for i in name:
        print(i)
        test = PygDataset(name = i)
        print(test)
        print(test[0])
        print([test[i].x[0] for i in range(100)])
        print(test[0].y)
        print(test[0].edge_index)
    
    
#     split_index = pyg_dataset.get_idx_split()
#     print(pyg_dataset[split_index['train']])
#     print(pyg_dataset[split_index['valid']])
#     print(pyg_dataset[split_index['test']])
    