from torch_geometric.datasets import Amazon, Coauthor, Flickr
import numpy as np
import torch
from .ppi import PPI
from .reddit import Reddit


def load(dataset):
    if dataset in ['ppi']:
        train_dataset = PPI('data/'+str(dataset), split='train').data
        train_feat = train_dataset.x
        train_edge_index = train_dataset.edge_index
        train_labels = train_dataset.y

        val_dataset = PPI('data/'+str(dataset), split='val').data
        val_feat = val_dataset.x
        val_edge_index = val_dataset.edge_index
        val_labels = val_dataset.y

        test_dataset = PPI('data/'+str(dataset), split='test').data
        test_feat = test_dataset.x
        test_edge_index = test_dataset.edge_index
        test_labels = test_dataset.y

        return train_edge_index, train_feat, train_labels, test_edge_index, test_feat, test_labels


    elif dataset in ['flickr']:
        full_dataset = Flickr('data/'+str(dataset)).data
        feat = full_dataset.x
        labels = full_dataset.y
        edge_index = full_dataset.edge_index

        train_mask = full_dataset.train_mask
        val_mask = full_dataset.val_mask
        test_mask = full_dataset.test_mask

        idx_train = np.argwhere(train_mask == 1).view(-1)
        idx_val = np.argwhere(val_mask == 1).view(-1)
        idx_test = np.argwhere(test_mask == 1).view(-1)

        return edge_index, feat, labels, idx_train, idx_val, idx_test

    elif dataset in ['reddit']:
        full_dataset = Reddit('data/'+str(dataset)).data
        feat = full_dataset.x
        labels = full_dataset.y
        edge_index = full_dataset.edge_index

        train_mask = full_dataset.train_mask
        val_mask = full_dataset.val_mask
        test_mask = full_dataset.test_mask

        idx_train = np.argwhere(train_mask == 1).view(-1)
        idx_val = np.argwhere(val_mask == 1).view(-1)
        idx_test = np.argwhere(test_mask == 1).view(-1)

        return edge_index, feat, labels, idx_train, idx_val, idx_test

    elif dataset in ['Computers', 'Photo']:
        full_dataset = Amazon('data/'+str(dataset), dataset).data

    elif dataset in ['CS', 'Physics']:
        full_dataset = Coauthor('data/'+str(dataset), dataset).data

    feat = full_dataset.x
    num_nodes = feat.shape[0]
    labels = full_dataset.y
    edge_index = full_dataset.edge_index

    idxs_train = []
    idxs_val = []
    idxs_test = []
    for i in range(20):
        np.random.seed(i)
        idx_test = torch.tensor(np.random.choice(num_nodes, int(num_nodes*0.8), replace=False)).reshape(1, -1)
        val_train = [i for i in np.arange(num_nodes) if i not in idx_test]
        np.random.seed(i)
        idx_val = torch.tensor(np.random.choice(val_train, int(num_nodes*0.1), replace=False)).reshape(1, -1)
        idx_train = torch.tensor([i for i in val_train if i not in idx_val]).reshape(1, -1)
        idxs_train.append(idx_train)
        idxs_val.append(idx_val)
        idxs_test.append(idx_test)

    idxs_train = torch.cat(idxs_train, 0)
    idxs_val = torch.cat(idxs_val, 0)
    idxs_test = torch.cat(idxs_test, 0)

    return edge_index, feat, labels, idxs_train, idxs_val, idxs_test


