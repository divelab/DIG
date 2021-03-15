import csv
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset
import networkx as nx


# get smiles from moses.csv
def get_smiles_moses(path):
    smile_list = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        for row in reader:
            smile = row[columns[0]]
            smile_list.append(smile)
    return smile_list

# get smiles and properties from zinc_800_graphaf.csv or zinc_800_jt.csv
def get_smiles_props_800(path):
    smile_list, prop_list = [], []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        for row in reader:
            smile = row[columns[0]]
            prop = row[columns[1]]
            smile_list.append(smile)
            prop_list.append(float(prop))
    return smile_list, prop_list

# get smiles from complete zinc250k.csv
def get_smiles_zinc250k(path):
    smile_list = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        for row in reader:
            smile = row[columns[1]]
            smile_list.append(smile)
    return smile_list

# get smiles and properties from complete zinc250k_property.csv or qm9_property.csv
def get_smiles_props_zinc250k_qm9(path, prop_name='qed'):
    smile_list, prop_list = [], []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        if prop_name == 'qed':
            prop_col = 0
        else:
            prop_col = 1
        for row in reader:
            smile = row[columns[1]]
            prop = row[columns[prop_col]]
            smile_list.append(smile)
            prop_list.append(float(prop))
    return smile_list, prop_list

# get smiles from complete qm9.csv
def get_smiles_qm9(path):
    smile_list = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        for row in reader:
            smile = row[columns[1]]
            smile_list.append(smile)
    return smile_list


class MolSet(Dataset):
    def __init__(self, smile_list, use_aug=False, prop_list=None, num_max_node=38, num_bond_type=3, edge_unroll=12, atom_list=[6, 7, 8, 9, 15, 16, 17, 35, 53]):
        """
        Pytorch data interface for molecule datasets.
        param smile_list: list of molecules in the form of SMILES strings
        param use_aug: if specified to be True, the order of atoms in molecules will be randomly shuffled
        param prop_list: list of molecular properties
        param num_max_node: the maximum number of nodes (atoms) among all molecules
        param num_bond_type: the number of edge (bond) types
        param edge_unroll: the maximum number of edges considered in a single breadth first search
        param atom_list: the list of atomic numbers of all atom types
        """
        self.atom_list = atom_list
        self.num_node_type = len(self.atom_list)
        self.num_bond_type = num_bond_type
        self.num_max_node = num_max_node
        self.node_features, self.adj_features, self.mol_sizes, self.n_mol = self._load_data(smile_list)
        self.all_smiles = smile_list
        if prop_list is not None:
            self.prop_list = prop_list
        self.edge_unroll = edge_unroll
        self.use_aug = use_aug


    def _load_data(self, smile_list):
        all_node_feature, all_adj_feature, all_mol_size = [], [], []
        valid_cnt = 0
        for smile in smile_list:
            atom_array, adj_array, mol_size = self._smile_process(smile)
            if atom_array is not None:
                valid_cnt += 1
                all_node_feature.append(atom_array)
                all_adj_feature.append(adj_array)
                all_mol_size.append(mol_size)

        print('total number of valid_molecule in dataset: {}'.format(valid_cnt))
        return np.array(all_node_feature), np.array(all_adj_feature), np.array(all_mol_size), valid_cnt


    def _smile_process(self, smile):
        mol = Chem.MolFromSmiles(smile)
        Chem.Kekulize(mol)
        num_atom = mol.GetNumAtoms()
        if num_atom > self.num_max_node:
            return None, None, None
        else:
            atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
            atom_array = np.zeros(self.num_max_node, dtype=np.int32)
            atom_array[:num_atom] = np.array(atom_list, dtype=np.int32)

            adj_array = np.zeros([self.num_bond_type, self.num_max_node, self.num_max_node], dtype=np.float32)
            bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                ch = bond_type_to_int[bond_type]
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                adj_array[ch, i, j] = 1.0
                adj_array[ch, j, i] = 1.0

        return atom_array, adj_array, num_atom


    def __len__(self):
        return self.n_mol


    def __getitem__(self, idx):
        node_feature, adj_feature, mol_size = self.node_features[idx].copy(), self.adj_features[idx].copy(), self.mol_sizes[idx].copy()

        pure_adj = np.sum(adj_feature, axis=0)[:mol_size, :mol_size]

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

        node_feature = node_feature[np.ix_(bfs_perm_origin)]
        adj_perm_index = np.ix_(bfs_perm_origin, bfs_perm_origin)
        for i in range(self.num_bond_type):
            adj_feature[i] = adj_feature[i][adj_perm_index]

        node_feature_one_hot = np.zeros([self.num_max_node, self.num_node_type], dtype=np.float32)
        for i in range(self.num_max_node):
            if node_feature[i] > 0:
                index = self.atom_list.index(node_feature[i])
                node_feature_one_hot[i, index] = 1.0
            
        adj_feature_extra_c = np.concatenate([adj_feature, 1 - np.sum(adj_feature, axis=0, keepdims=True)], axis=0)
        for i in range(self.num_bond_type + 1):
            adj_feature_extra_c[i] += np.eye(self.num_max_node)
        
        raw_smile = self.all_smiles[idx]

        if hasattr(self, 'prop_list'):
            prop = self.prop_list[idx]
            return {'node': torch.Tensor(node_feature_one_hot), 'adj': torch.Tensor(adj_feature_extra_c), 'mol_size': mol_size, 
                'bfs_perm_origin': torch.Tensor(bfs_perm_origin), 'raw_smile': raw_smile, 'prop': prop}
        else:
            return {'node': torch.Tensor(node_feature_one_hot), 'adj': torch.Tensor(adj_feature_extra_c), 'mol_size': mol_size, 
                'bfs_perm_origin': torch.Tensor(bfs_perm_origin), 'raw_smile': raw_smile}


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
