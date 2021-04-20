import csv
import numpy as np
from rdkit import Chem
import os, shutil, re, torch, json, ast
import os.path as osp
import pandas as pd
# import scipy.sparse as sp
from torch.utils.data import Dataset
import networkx as nx
import ssl
from six.moves import urllib

from itertools import repeat, product
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import to_networkx

bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
zinc_atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
qm9_atom_list = [6, 7, 8, 9]

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
        atomic_num_idx = atom_features[local_perm[i]]
        G.add_node(i, atom_num_idx=atomic_num_idx)
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
                 root='./',
                 name='qm9',
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
        
        self.processed_filename = processed_filename
        self.root = root
        self.name = name
#         self.edge_unroll = edge_unroll    # not used
        self.prop_name = prop_name
        self.use_aug = use_aug
        self.one_shot = one_shot
        
        if conf_dict is None:                        
            config_file = pd.read_csv(os.path.join(os.path.dirname(__file__), 'config.csv'), index_col = 0)
            if not self.name in config_file:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(config_file.keys())
                raise ValueError(error_mssg)
            config = config_file[self.name]
            
        else:
            config = conf_dict
            
        self.url = config['url']
        self.available_prop = str(prop_name) in ast.literal_eval(config['prop_list'])
        self.smile_col = config['smile']
        self.num_max_node = int(config['num_max_node'])
        self.atom_list = ast.literal_eval(config['atom_list'])
            
        super(PygDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not osp.exists(self.raw_paths[0]):
            self.download()
        if osp.exists(self.processed_paths[0]):
            self.data, self.slices, self.all_smiles = torch.load(self.processed_paths[0])
        else:
            self.process()

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
    def raw_dir(self):
        name = 'raw'
#         return osp.join(self.root, self.name, name)
        return osp.join(self.root, name)
    
    @property
    def processed_dir(self):
        name = 'processed'
        if self.one_shot:
            name = 'processed_'+'oneshot'
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
    
    def download(self):
        print('making raw files:', self.raw_dir)
        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        url = self.url
        path = download_url(url, self.raw_dir)
#         extract_zip(path, folder)
#         os.unlink(path)
#         shutil.rmtree(self.raw_dir)
#         os.rename(osp.join(folder), self.raw_dir)

    def process(self):
        print('Processing...')
        if self.one_shot:
            self.data, self.slices = self.one_hot_process()
        else:
            self.data, self.slices = self.pre_process()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        torch.save((self.data, self.slices, self.all_smiles), self.processed_paths[0])
        print('Done!')
        
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
        
        if not self.one_shot:
            # bfs-searching order
            mol_size = data.num_atom.numpy()[0]
            pure_adj = np.sum(data.adj[:3].numpy(), axis=0)[:mol_size, :mol_size]
            if self.use_aug:
                local_perm = np.random.permutation(mol_size)
                adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
                G = nx.from_numpy_matrix(np.asmatrix(adj_perm))
                start_idx = np.random.randint(adj_perm.shape[0])
            else:
                local_perm = np.arange(mol_size)
                G = nx.from_numpy_matrix(np.asmatrix(pure_adj))
                start_idx = 0

            bfs_perm = np.array(self._bfs_seq(G, start_idx))
            bfs_perm_origin = local_perm[bfs_perm]
            bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.num_max_node)])
            data.x = data.x[bfs_perm_origin]
            for i in range(4):
                data.adj[i] = data.adj[i][bfs_perm_origin][:,bfs_perm_origin]
            
            data['bfs_perm_origin'] = torch.Tensor(bfs_perm_origin).long()

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
                atom_array = np.zeros((self.num_max_node, len(self.atom_list)), dtype=np.float32)

                atom_idx = 0
                for atom in mol.GetAtoms():
                    atom_feature = atom.GetAtomicNum()
                    atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1
                    atom_idx += 1
                    
                x = torch.tensor(atom_array)

                # bonds
                adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    ch = bond_type_to_int[bond_type]
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    adj_array[ch, i, j] = 1.0
                    adj_array[ch, j, i] = 1.0
                adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
                adj_array += np.eye(self.num_max_node)

                data = Data(x=x)
                data.adj = torch.tensor(adj_array)
                data.num_atom = num_atom
                if self.available_prop:
                    data.y = torch.tensor([float(prop_list[i])])
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices
    
    def one_hot_process(self):
        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=',', dtype='str')
        smile_list = list(input_df[self.smile_col])
        if self.available_prop:
                prop_list = list(input_df[self.prop_name])
                
        self.all_smiles = smile_list
        data_list = []
        
#         atom_list = ast.literal_eval(self.atom_list)
        
        for i in range(len(smile_list)):
            smile = smile_list[i]
            mol = Chem.MolFromSmiles(smile)
            Chem.Kekulize(mol)
            num_atom = mol.GetNumAtoms()
            if num_atom > self.num_max_node:
                continue
            else:
                # atoms
                atom_array = np.zeros((len(self.atom_list), self.num_max_node), dtype=np.int32)
                if self.one_shot:
                    virtual_node = np.ones((1, self.num_max_node), dtype=np.int32)

                atom_idx = 0
                for atom in mol.GetAtoms():
                    atom_feature = atom.GetAtomicNum()
#                     print('self.atom_list','atom_feature', 'index')
#                     print(self.atom_list, atom_feature, self.atom_list.index(atom_feature))
                    atom_array[self.atom_list.index(atom_feature), atom_idx] = 1
                    if self.one_shot:
                        virtual_node[0, atom_idx] = 0
                    atom_idx += 1
                    
                if self.one_shot:
                    x = torch.tensor(np.concatenate((atom_array, virtual_node), axis=0))
                else:
                    x = torch.tensor(atom_array)

                # bonds
                adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    ch = bond_type_to_int[bond_type]
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    adj_array[ch, i, j] = 1.0
                    adj_array[ch, j, i] = 1.0
                adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
                                
                data = Data(x=x)
                data.adj = torch.tensor(adj_array)
                data.num_atom = num_atom
                if self.available_prop:
                    data.y = torch.tensor([float(prop_list[i])])
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices
    
    def get_split_idx(self):
        if self.name.find('zinc250k') != -1:
            if not osp.exists('./raw/valid_idx_zinc250k.json'):
                path = './raw/valid_idx_zinc250k.json'
                url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/valid_idx_zinc250k.json'
                context = ssl._create_unverified_context()
                data = urllib.request.urlopen(url, context=context)
                with open(path, 'wb') as f:
                    f.write(data.read())
            with open('./raw/valid_idx_zinc250k.json') as f:
                valid_idx = json.load(f)
                
        elif self.name.find('qm9') != -1:
            if not osp.exists('./raw/valid_idx_qm9.json'):
                path = './raw/valid_idx_qm9.json'
                url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/valid_idx_qm9.json'
                context = ssl._create_unverified_context()
                data = urllib.request.urlopen(url, context=context)
                with open(path, 'wb') as f:
                    f.write(data.read())
            with open('./raw/valid_idx_qm9.json') as f:
                valid_idx = json.load(f)['valid_idxs']
                valid_idx = list(map(int, valid_idx))
        else:
            print('No available split file for this dataset, please check.')
            return None
        
        train_idx = list(set(np.arange(self.__len__())).difference(set(valid_idx)))

        return {'train_idx': torch.tensor(train_idx, dtype = torch.long), 'valid_idx': torch.tensor(valid_idx, dtype = torch.long)}
    
    def _bfs_seq(self, G, start_id):
        dictionary = dict(nx.bfs_successors(G, start_id))
        start = [start_id]
        output = [start_id]
        while len(start) > 0:
            next = []
            while len(start) > 0:
                current = start.pop(0)
                neighbor = dictionary.get(current)
                if neighbor is not None:
                    next = next + neighbor
            output = output + next
            start = next
        return output
    
if __name__ == '__main__':
    name = pd.read_csv('config.csv', index_col = 0)
    for i in name:
        test = PygDataset(name = i)
        print(test)
        print(test[0])
        print(test[0].y)
    
    
#     atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
#             atom_array = np.zeros(self.num_max_node, dtype=np.int32)
#             atom_array[:num_atom] = np.array(atom_list, dtype=np.int32)

#             adj_array = np.zeros([self.num_bond_type, self.num_max_node, self.num_max_node], dtype=np.float32)
#             bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
#             for bond in mol.GetBonds():
#                 bond_type = bond.GetBondType()
#                 ch = bond_type_to_int[bond_type]
#                 i = bond.GetBeginAtomIdx()
#                 j = bond.GetEndAtomIdx()
#                 adj_array[ch, i, j] = 1.0
#                 adj_array[ch, j, i] = 1.0