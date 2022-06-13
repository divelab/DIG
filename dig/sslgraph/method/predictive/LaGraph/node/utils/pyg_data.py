from torch_geometric.datasets import WikiCS, Amazon, Coauthor, Planetoid, Flickr, Yelp, AMiner
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
from .ppi import PPI
from .reddit import Reddit
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def load(dataset):
    if dataset in ['wikics']:
        full_dataset = WikiCS('data/'+str(dataset)).data
        feat = full_dataset.x
        num_nodes = feat.shape[0]
        labels = full_dataset.y
        edge_index = full_dataset.edge_index
        # adj = to_scipy_sparse_matrix(edge_index=edge_index)

        train_mask = full_dataset.train_mask
        val_mask = full_dataset.val_mask
        test_mask = full_dataset.test_mask
        n_splits = train_mask.shape[1]
        idxs_train = torch.cat([np.argwhere(train_mask[:, i]).reshape(1, -1) for i in range(n_splits)], 0)
        idxs_val = torch.cat([np.argwhere(val_mask[:, i]).reshape(1, -1) for i in range(n_splits)], 0)
        idxs_test = torch.cat([np.argwhere(test_mask == 1).reshape(1, -1) for i in range(n_splits)], 0)

        return edge_index, feat, labels, idxs_train, idxs_val, idxs_test

    elif dataset in ['PubMed', 'pubmed', 'Cora', 'CiteSeer']:
        full_dataset = Planetoid('data/'+str(dataset), dataset, split='public').data
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

    elif dataset in ['ppi']:
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

        # n_train = train_feat.shape[0]
        # n_val = val_feat.shape[0]
        # n_test = test_feat.shape[0]
        # val_edge_index += n_train
        # test_edge_index += (n_train + n_val)
        # edge_index = torch.cat((train_edge_index, val_edge_index, test_edge_index), dim=1)
        #
        # feat = torch.cat((train_feat, val_feat, test_feat))
        # labels = torch.cat((train_labels, val_labels, test_labels)).long()  # (56944, 121)
        # idx_train = list(range(n_train))
        # idx_val = list(range(n_val))
        # idx_test = list(range(n_test))
        #
        # return edge_index, feat, labels, idx_train, idx_val, idx_test

    elif dataset in ['aminer']:
        full_dataset = AMiner('data/'+str(dataset)).data
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

    elif dataset in ['yelp']:
        full_dataset = Yelp('data/'+str(dataset)).data
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

    elif dataset in ['arxiv']:
        dataset_class = PygNodePropPredDataset(name='ogbn-arxiv', root='data/arxiv')
        full_dataset = dataset_class[0]
        feat = full_dataset.x
        labels = full_dataset.y.view(-1)
        edge_index = to_undirected(full_dataset.edge_index, full_dataset.num_nodes)

        split_idx = dataset_class.get_idx_split()
        idx_train = split_idx['train']
        idx_val = split_idx['valid']
        idx_test = split_idx['test']

        return edge_index, feat, labels, idx_train, idx_val, idx_test

    elif dataset in ['Computers', 'Photo']:
        full_dataset = Amazon('data/'+str(dataset), dataset).data

    elif dataset in ['CS', 'Physics']:
        full_dataset = Coauthor('data/'+str(dataset), dataset).data

    feat = full_dataset.x
    num_nodes = feat.shape[0]
    labels = full_dataset.y
    edge_index = full_dataset.edge_index
    # adj = to_scipy_sparse_matrix(edge_index=edge_index)

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


def edge_perturbation(edge_index, node_num, add=True, drop=False, ratio=0.1):
    '''
    Args:
        add (bool): Set True if randomly add edges in a given graph.
        drop (bool): Set True if randomly drop edges in a given graph.
        ratio: Percentage of edges to add or drop.
    '''

    _, edge_num = edge_index.size()
    perturb_num = int(edge_num * ratio)

    edge_index = edge_index.detach().clone()
    idx_remain = edge_index
    idx_add = torch.tensor([]).reshape(2, -1).long()

    if drop:
        idx_remain = edge_index[:, np.random.choice(edge_num, edge_num-perturb_num, replace=False)]

    if add:
        idx_add = torch.randint(node_num, (2, perturb_num))

    new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
    new_edge_index = torch.unique(new_edge_index, dim=1)

    return new_edge_index


def node_sample(edge_index, features, sample_size=100):
    '''
    Args:
        sample_size: Number of nodes to sample.
    '''

    node_num = features.shape[1]
    _, edge_num = edge_index.size()

    idx_nondrop = np.random.choice(node_num, sample_size, replace=False)
    # idx_drop = [n for n in range(node_num) if not n in idx_nondrop]
    adj = to_dense_adj(edge_index, max_num_nodes=node_num)[0]
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    # sample_edge_index = dense_to_sparse(adj)[0]
    adj = np.array(adj).reshape(sample_size, sample_size)

    sample_feat = features[:, idx_nondrop]
    print('sample_feat.shape:', sample_feat.shape)

    return adj, sample_feat


def node_sample_index(edge_index, features, sample_size=100):
    '''
    Args:
        sample_size: Number of nodes to sample.
    '''

    node_num = features.shape[0]
    _, edge_num = edge_index.size()

    idx_nondrop = np.random.choice(node_num, sample_size, replace=False)
    # idx_drop = [n for n in range(node_num) if not n in idx_nondrop]
    adj = to_dense_adj(edge_index, max_num_nodes=node_num)[0]
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    sample_edge_index = dense_to_sparse(adj)[0]
    # adj = np.array(adj).reshape(sample_size, sample_size)

    sample_feat = features[idx_nondrop]
    print('sample_feat.shape:', sample_feat.shape)

    return sample_edge_index, sample_feat