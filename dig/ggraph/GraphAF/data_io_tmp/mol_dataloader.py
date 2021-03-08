import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from torch.utils.data import Dataset
import networkx as nx

class MolSet(Dataset):
    def __init__(self, smile_list, mode='pretrain', prop_list=None, num_max_node=38, num_bond_type=3, edge_unroll=12, atom_list=[6, 7, 8, 9, 15, 16, 17, 35, 53]):
        self.atom_list = atom_list
        self.num_node_type = len(self.atom_list)
        self.num_bond_type = num_bond_type
        self.num_max_node = num_max_node
        self.node_features, self.adj_features, self.mol_sizes, self.n_mol = self._load_data(smile_list)
        self.all_smiles = smile_list
        self.mode = mode
        if mode == 'rl_ft':
            assert prop_list is not None
            self.prop_list = prop_list
        self.edge_unroll = edge_unroll


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
            return None, None, None,None
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
        local_perm = np.random.permutation(mol_size)
        adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
        G = nx.from_numpy_matrix(np.asmatrix(adj_perm))

        start_idx = np.random.randint(adj_perm.shape[0])
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
            
        # adj_feature_extra_c = np.concatenate([adj_feature, 1 - np.sum(adj_feature, axis=0, keepdims=True)], axis=0)
        adj_feature_extra_c = np.concatenate([adj_feature, np.zeros_like(adj_feature[0:1])], axis=0)
        adj_feature_extra_c[-1, :mol_size, :mol_size] = 1 - pure_adj
        for i in range(self.num_bond_type + 1):
            adj_feature_extra_c[i] += np.eye(self.num_max_node)

        if mol_size < self.edge_unroll:
            total_edge = int(mol_size * (mol_size - 1) / 2)
        else:
            total_edge = int(self.edge_unroll * (self.edge_unroll - 1) / 2 + self.edge_unroll * (mol_size - self.edge_unroll))
        
        num_max_edge = int(self.edge_unroll * (self.edge_unroll - 1) / 2 + self.edge_unroll * (self.num_max_node - self.edge_unroll))
        node_mask = torch.zeros([self.num_max_node])
        node_mask[:mol_size] = 1
        edge_mask = torch.zeros([num_max_edge])
        edge_mask[:total_edge] = 1

        if self.mode == 'pretrain':
            return {'node': torch.Tensor(node_feature_one_hot), 'adj': torch.Tensor(adj_feature_extra_c), 'node_mask': node_mask, 'edge_mask': edge_mask}
        elif self.mode == 'rl_ft':
            prop = self.prop_list[idx]
            raw_smile = self.all_smiles[idx]
            return {'node': torch.Tensor(node_feature_one_hot), 'adj': torch.Tensor(adj_feature_extra_c), 'raw_smile': raw_smile,
                'prop': prop, 'mol_size': mol_size, 'bfs_perm_origin': bfs_perm_origin}


    def _bfs_seq(self, G, start_id):
        '''
        get a bfs node sequence
        :param G:
        :param start_id:
        :return:
        '''
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