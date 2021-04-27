import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data


class PygQM9Dataset(InMemoryDataset):
    def __init__(self, root = 'dataset/qm9', prop_name = 'U0'):
        '''
            Pytorch Geometric QM9 dataset object
                - root (str): the dataset folder will be located at root
        '''

        self.prop_name = prop_name
        self.url = 'https://github.com/klicperajo/dimenet/blob/master/data/qm9_eV.npz'


        super(PygQM9Dataset, self).__init__(root, prop_name)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'qm9_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        
        data = np.load('/mnt/dive/shared/limei/DIG/dataset/qm9/raw/qm9_eV.npz',allow_pickle=True) #(osp.join(self.raw_dir, self.raw_file_names))

        R = data['R']
        Z = data['Z']
        N= data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z,split)
        y = np.expand_dims(data[self.prop_name],axis=-1)

        data_list = []
        for i in tqdm(range(len(y))):
            R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            y_i = torch.tensor(y[i],dtype=torch.float32)
            data = Data(pos=R_i, z=z_i, y=y_i)

            data_list.append(data)

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        ids = shuffle(130831, random_state=42)
        train_size = 110000
        valid_size = 10000
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = PygQM9Dataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())