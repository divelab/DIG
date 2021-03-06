import torch
import numpy as np
from sklearn.utils import shuffle
from torch_geometric.data import DataLoader, Data


def load_md17(dataset, train_size, valid_size):
    data = np.load('../datasets/' + dataset + '_dft.npz')
    E = data['E']
    F = data['F']
    R = data['R']
    z = data['z']
    num_atom = len(z)
    num_molecule = len(E)
    dataset = []
    for i in range(num_molecule):
        R_i = torch.tensor(R[i],dtype=torch.float32)
        z_i = torch.tensor(z,dtype=torch.int64)
        E_i = torch.tensor(E[i],dtype=torch.float32)
        F_i = torch.tensor(F[i],dtype=torch.float32)
        data = Data(pos=R_i, z=z_i, y=E_i, force=F_i)
        dataset.append(data)
    ids = shuffle(range(num_molecule), random_state=42)
    train_idx, val_idx, test_idx = np.array(ids[:train_size]), np.array(ids[train_size:train_size + valid_size]), np.array(ids[train_size + valid_size:])

    train_dataset = [dataset[int(i)] for i in train_idx]
    val_dataset = [dataset[int(i)] for i in val_idx]
    test_dataset = [dataset[int(i)] for i in test_idx]
    return train_dataset, val_dataset, test_dataset, num_atom


def load_qm9(dataset, target, train_size, valid_size):
    data = np.load('../datasets/' + dataset + '_eV.npz')
    R = data['R']
    Z = data['Z']
    N= data['N']
    split = np.cumsum(N)
    R_qm9 = np.split(R, split)
    Z_qm9 = np.split(Z,split)
    y = np.expand_dims(data[target],axis=-1)
    num_molecule = len(y)
    dataset = []
    for i in range(num_molecule):
        R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
        z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
        y_i = torch.tensor(y[i],dtype=torch.float32)
        data = Data(pos=R_i, z=z_i, y=y_i)
        dataset.append(data)
    ids = shuffle(range(num_molecule), random_state=42)
    train_idx, val_idx, test_idx = np.array(ids[:train_size]), np.array(ids[train_size:train_size + valid_size]), np.array(ids[train_size + valid_size:])

    train_dataset = [dataset[int(i)] for i in train_idx]
    val_dataset = [dataset[int(i)] for i in val_idx]
    test_dataset = [dataset[int(i)] for i in test_idx]
    return train_dataset, val_dataset, test_dataset