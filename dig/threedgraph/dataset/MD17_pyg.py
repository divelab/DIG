import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data


class PygMD17Dataset(InMemoryDataset):
    def __init__(self, name = 'benzene'):
        '''
            Pytorch Geometric QM9 dataset object
                - root (str): the dataset folder will be located at root
        '''

        self.name = name
        self.root = 'dataset/' + self.name
        self.url = 'http://quantum-machine.org/gdml/' + self.name + '_dft.npz'


        super(PygMD17Dataset, self).__init__(name)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.name + '_dft.npz'

    @property
    def processed_file_names(self):
        return self.name + '_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        E = data['E']
        F = data['F']
        R = data['R']
        z = data['z']
        self.num_molecules = len(E)

        data_list = []
        for i in tqdm(range(len(E))):
            R_i = torch.tensor(R[i],dtype=torch.float32)
            z_i = torch.tensor(z,dtype=torch.int64)
            E_i = torch.tensor(E[i],dtype=torch.float32)
            F_i = torch.tensor(F[i],dtype=torch.float32)
            data = Data(pos=R_i, z=z_i, y=E_i, force=F_i)

            data_list.append(data)

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        ids = shuffle(self.num_molecules, random_state=42)
        train_size = 1000
        valid_size = 10000
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = PygMD17Dataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())