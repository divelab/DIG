import os
import os.path as osp
import numpy as np
import torch
import networkx as nx
from networkx.algorithms import tree
from math import pi
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_tar



atomic_num_to_type={1:0, 6:1, 7:2, 8:3, 9:4}
bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}


def collate_fn(data_batch_list):
    r"""
        The collate function used to form a mini-batch of data when creating data loaders with the dataset class dig.ggraph3D.dataset.QM93DGEN.

        Args:
            data_batch_list: a list of python dict returned from the get function of dig.ggraph3D.dataset.QM93DGEN.
        :rtype: :class:`dict` a python dict with the same keys as every python dict in data_batch_list.
    """
    data_batch = {}

    for key in ['atom_type', 'position', 'new_atom_type', 'new_dist', 'new_angle', 'new_torsion', 'cannot_focus']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in data_batch_list], dim=0)
    
    num_steps_list = torch.tensor([0]+[len(data_batch_list[i]['new_atom_type']) for i in range(len(data_batch_list)-1)])
    batch_idx_offsets = torch.cumsum(num_steps_list, dim=0)
    repeats = torch.tensor([len(mol_dict['batch']) for mol_dict in data_batch_list])
    batch_idx_repeated_offsets = torch.repeat_interleave(batch_idx_offsets, repeats)
    batch_offseted = torch.cat([mol_dict['batch'] for mol_dict in data_batch_list], dim=0) + batch_idx_repeated_offsets
    data_batch['batch'] = batch_offseted

    num_atoms_list = torch.tensor([0]+[len(data_batch_list[i]['atom_type']) for i in range(len(data_batch_list)-1)])
    atom_idx_offsets = torch.cumsum(num_atoms_list, dim=0)
    for key in ['focus', 'c1_focus', 'c2_c1_focus']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in data_batch_list])
        atom_idx_repeated_offsets = torch.repeat_interleave(atom_idx_offsets, repeats)
        atom_offseted = torch.cat([mol_dict[key] for mol_dict in data_batch_list], dim=0) + atom_idx_repeated_offsets[:,None]
        data_batch[key] = atom_offseted

    return data_batch


class QM93DGEN(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for datasets used in molecule generation.
        
        .. note::
            When creating data loaders of this dataset class, only dig.ggraph3D.dataset.collate_fn can be used as the collate function.
        
        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            subset_idxs (list, optional): if it is not None, only the data located 
                at the indexs in subset_idxs of the dataset will be sampled.
                (default: :obj:`None`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
    """
    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
    split_urls = [
        "https://github.com/divelab/DIG_storage/raw/main/ggraph3D/data/split.npz",
        "https://github.com/divelab/DIG_storage/raw/main/ggraph3D/data/alpha.npz",
        "https://github.com/divelab/DIG_storage/raw/main/ggraph3D/data/gap.npz"
    ]
    
    def __init__(self, root='./qm9_3Dgen', subset_idxs=None, transform=None, pre_transform=None, pre_filter=None):
        super(QM93DGEN, self).__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        if not osp.exists(self.raw_paths[0]):
            self.download()
        if osp.exists(self.processed_paths[0]):
            self.atom_type_list, self.position_list, self.con_mat_list = torch.load(self.processed_paths[0])
        else:
            self.process()
        self._indices = subset_idxs if subset_idxs is not None else range(len(self.atom_type_list))
    
    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        return 'gdb9.sdf'
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    

    def download(self):
        print('making raw files:', self.raw_dir)
        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        path = download_url(self.raw_url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)
        
        for split_url in self.split_urls:
            download_url(split_url, self.raw_dir)
    

    def process(self):
        r"""
            Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        """
        mols = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        self.atom_type_list, self.position_list, self.con_mat_list = [], [], []

        for idx in range(len(mols)):
            mol = mols[idx]
            num_atoms = mol.GetNumAtoms()
            position = mols.GetItemText(idx).split('\n')[4:4+num_atoms]
            position = np.array([[float(x) for x in line.split()[:3]] for line in position], dtype=np.float32)
            atom_type = np.array([atomic_num_to_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()])

            con_mat = np.zeros([num_atoms, num_atoms], dtype=int)
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = bond_to_type[bond.GetBondType()]
                con_mat[start, end] = bond_type
                con_mat[end, start] = bond_type
            
            if atom_type[0] != 1:
                carbon_idxs = np.nonzero(atom_type == 1)
                perm = np.arange(len(atom_type))
                if len(carbon_idxs[0]) > 0:
                    first_carbon = carbon_idxs[0][0]
                    perm[0] = first_carbon
                    perm[first_carbon] = 0
                atom_type, position = atom_type[perm], position[perm]
                con_mat = con_mat[perm][:, perm]
            
            self.atom_type_list.append(torch.tensor(atom_type))
            self.position_list.append(torch.tensor(position))
            self.con_mat_list.append(torch.tensor(con_mat))
        
        torch.save((self.atom_type_list, self.position_list, self.con_mat_list), self.processed_paths[0])
    
    
    def len(self):
        r"""
            Gets the number of molecular geometries that can be sampled in total.
        """
        return len(self._indices)
    

    def get_idx_split(self, task):
        r"""Gets the train-valid set split indices of the dataset for different tasks.
        
        Args:
            task: The name of the task that the dataset will be used in, including 'rand_gen' for random molecular geometry generation, 
                'gap_opt' for discovering molecular geometries with low HOMO-LUMO gaps, and 'alpha_opt' for discovering molecular geometries 
                with high isotropic polarizabilities.
        :rtype: A dictionary for training-validation split with key :obj:`train` and :obj:`valid`.
        """
        assert task in ['rand_gen', 'gap_opt', 'alpha_opt']
        if task == 'rand_gen':
            split_idxs = np.load(osp.join(self.raw_dir, 'split.npz'))
        elif task == 'gap_opt':
            split_idxs = np.load(osp.join(self.raw_dir, 'gap.npz'))
        elif task == 'alpha_opt':
            split_idxs = np.load(osp.join(self.raw_dir, 'alpha.npz'))
        else:
            print("Not supported task! The task must be rand_gen, gap_opt, or alpha_opt!")
            exit()
        split_dict = {
            'train': split_idxs['train_idx'].tolist(), 
            'valid': split_idxs['val_idx'].tolist()
        }
        return split_dict


    def get(self, idx):
        r"""Gets the data object at index :idx:.
        
        Args:
            idx: The index of the data that you want to reach.
        :rtype: :class:`dict` a python dict with the following items:
                    "atom_type" --- the atom types of previously generated geometries at each generation step;
                    "position" --- the atom coordinates of previously generated geometries at each generation step;
                    "batch" --- the identity indexes used to discriminate different generation step;
                    "focus" --- the index of focus atom at each generation step;
                    "c1_focus" --- the index of c1 and focus atom at each generation step;
                    "c2_c1_focus" --- the index of c2, c1, and focus atom at each generation step;
                    "new_atom_type" --- the atom types to be generated at each generation step;
                    "new_dist" --- the distances to be generated at each generation step;
                    "new_angle" --- the angles to be generated at each generation step;
                    "new_torsion" --- the torsion angles to be generated at each generation step;
                    "cannot_focus" --- the labels denoting whether each atom can serve as the focus atom or not in the previously generated geometries at each generation step.
        """
        atom_type, position, con_mat = self.atom_type_list[idx], self.position_list[idx], self.con_mat_list[idx]
        atom_valency = torch.sum(con_mat, dim=1)

        squared_dist = torch.sum(torch.square(position[:,None,:] - position[None,:,:]), dim=-1)      
        nx_graph = nx.from_numpy_matrix(squared_dist.numpy())
        edges = list(tree.minimum_spanning_edges(nx_graph, algorithm='prim', data=False))

        focus_node_id, target_node_id = zip(*edges)
        # print(focus_node_id, target_node_id)

        node_perm = torch.cat((torch.tensor([0]), torch.tensor(target_node_id)))
        position = position[node_perm]
        atom_type = atom_type[node_perm]
        con_mat = con_mat[node_perm][:,node_perm]
        squared_dist = squared_dist[node_perm][:,node_perm]
        atom_valency = atom_valency[node_perm]
        # print(con_mat)s

        focus_node_id = torch.tensor(focus_node_id)
        steps_focus = torch.nonzero(focus_node_id[:,None] == node_perm[None,:])[:,1]
        steps_c1_focus, steps_c2_c1_focus = torch.empty([0,2], dtype=int), torch.empty([0,3], dtype=int)
        steps_batch, steps_position, steps_atom_type = torch.empty([0,1], dtype=int), torch.empty([0,3], dtype=position.dtype), torch.empty([0,1], dtype=atom_type.dtype)
        steps_cannot_focus = torch.empty([0,1], dtype=float)
        steps_dist, steps_angle, steps_torsion = torch.empty([0,1], dtype=float), torch.empty([0,1], dtype=float), torch.empty([0,1], dtype=float)
        idx_offsets = torch.cumsum(torch.arange(len(atom_type) - 1), dim=0)
        
        for i in range(len(atom_type) - 1):
            partial_con_mat = con_mat[:i+1, :i+1]
            valency_sum = partial_con_mat.sum(dim=1, keepdim=True)
            steps_cannot_focus = torch.cat((steps_cannot_focus, (valency_sum == atom_valency[:i+1, None]).float()))

            one_step_focus = steps_focus[i]
            focus_pos, new_pos = position[one_step_focus], position[i+1]
            one_step_dis = torch.norm(new_pos - focus_pos)
            steps_dist = torch.cat((steps_dist, one_step_dis.view(1,1)))
            
            if i > 0:
                mask = torch.ones([i+1], dtype=torch.bool)
                mask[one_step_focus] = False
                c1_dists = squared_dist[one_step_focus, :i+1][mask]
                one_step_c1 = torch.argmin(c1_dists)
                if one_step_c1 >= one_step_focus:
                    one_step_c1 += 1
                steps_c1_focus = torch.cat((steps_c1_focus, torch.tensor([one_step_c1, one_step_focus]).view(1,2) + idx_offsets[i]))

                c1_pos = position[one_step_c1]
                a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                one_step_angle = torch.atan2(b,a)
                steps_angle = torch.cat((steps_angle, one_step_angle.view(1,1)))

                if i > 1:
                    mask[one_step_c1] = False
                    c2_dists = squared_dist[one_step_c1, :i+1][mask]
                    one_step_c2 = torch.argmin(c2_dists)
                    if one_step_c2 >= min(one_step_c1, one_step_focus):
                        one_step_c2 += 1
                        if one_step_c2 >= max(one_step_c1, one_step_focus):
                            one_step_c2 += 1
                    steps_c2_c1_focus = torch.cat((steps_c2_c1_focus, torch.tensor([one_step_c2, one_step_c1, one_step_focus]).view(1,3) + idx_offsets[i]))

                    c2_pos = position[one_step_c2]
                    plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                    plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
                    b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                    one_step_torsion = torch.atan2(b, a)
                    steps_torsion = torch.cat((steps_torsion, one_step_torsion.view(1,1)))
                    
            one_step_position = position[:i+1]
            steps_position = torch.cat((steps_position, one_step_position), dim=0)
            one_step_atom_type = atom_type[:i+1]
            steps_atom_type = torch.cat((steps_atom_type, one_step_atom_type.view(-1,1)))
            steps_batch = torch.cat((steps_batch, torch.tensor([i]).repeat(i+1).view(-1,1)))
        
        steps_focus += idx_offsets
        steps_new_atom_type = atom_type[1:]
        steps_torsion[steps_torsion <= 0] += 2 * pi

        data_batch = {}
        data_batch['atom_type'] = steps_atom_type.view(-1)
        data_batch['position'] = steps_position
        data_batch['batch'] = steps_batch.view(-1)
        data_batch['focus'] = steps_focus[:,None]
        data_batch['c1_focus'] = steps_c1_focus
        data_batch['c2_c1_focus'] = steps_c2_c1_focus
        data_batch['new_atom_type'] = steps_new_atom_type.view(-1)
        data_batch['new_dist'] = steps_dist
        data_batch['new_angle'] = steps_angle
        data_batch['new_torsion'] = steps_torsion
        data_batch['cannot_focus'] = steps_cannot_focus.view(-1).float()

        return data_batch