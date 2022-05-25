import os
import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import files_exist
import shutil


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        edge_index = dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0]
        data_list.append(Data(x=torch.from_numpy(node_features[graph_idx]).float(),
                              edge_index=edge_index,
                              y=torch.from_numpy(np.where(graph_labels[graph_idx])[0])))
    return data_list


class SynGraphDataset(InMemoryDataset):
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.

    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'
    # Format: name: [display_name, url_name, filename]
    names = {
        'ba_shapes': ['BA_shapes', 'BA_shapes.pkl', 'BA_shapes'],
        'ba_community': ['BA_Community', 'BA_Community.pkl', 'BA_Community'],
        'tree_grid': ['Tree_Grid', 'Tree_Grid.pkl', 'Tree_Grid'],
        'tree_cycle': ['Tree_Cycle', 'Tree_Cycles.pkl', 'Tree_Cycles'],
        'ba_2motifs': ['BA_2Motifs', 'BA_2Motifs.pkl', 'BA_2Motifs']
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.pkl'

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)

    def process(self):
        if self.name.lower() == 'BA_2Motifs'.lower():
            data_list = read_ba2motif_data(self.raw_dir, self.names[self.name][2])

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [data for data in data_list if self.pre_filter(data)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)
        else:
            # Read data into huge `Data` list.
            data = self.read_syn_data()
            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list = [data]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

    def gen_motif_edge_mask(self, data, node_idx=0, num_hops=3):
        if self.name in ['ba_2motifs']:
            return torch.logical_and(data.edge_index[0] >= 20, data.edge_index[1] >= 20)
        elif self.name in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            """ selection in a loop way to fetch all the nodes in the connected motifs """
            if data.y[node_idx] == 0:
                return torch.zeros_like(data.edge_index[0]).type(torch.bool)
            connected_motif_nodes = set()
            edge_label_matrix = data.edge_label_matrix + data.edge_label_matrix.T
            edge_index = data.edge_index.to('cpu')
            if isinstance(node_idx, torch.Tensor):
                connected_motif_nodes.add(node_idx.item())
            else:
                connected_motif_nodes.add(node_idx)
            for _ in range(num_hops):
                append_node = set()
                for node in connected_motif_nodes:
                    append_node.update(tuple(torch.where(edge_label_matrix[node] != 0)[0].tolist()))
                connected_motif_nodes.update(append_node)
            connected_motif_nodes_tensor = torch.Tensor(list(connected_motif_nodes))
            frm_mask = (edge_index[0].unsqueeze(1) - connected_motif_nodes_tensor.unsqueeze(0) == 0).any(dim=1)
            to_mask = (edge_index[1].unsqueeze(1) - connected_motif_nodes_tensor.unsqueeze(0) == 0).any(dim=1)
            return torch.logical_and(frm_mask, to_mask)

    def read_syn_data(self):
        with open(self.raw_paths[0], 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

        x = torch.from_numpy(features).float()
        y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        y = torch.from_numpy(np.where(y)[1])
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)
        data.edge_label_matrix = torch.from_numpy(edge_label_matrix)
        return data


class BA_LRP(InMemoryDataset):
    r"""
    The synthetic graph classification dataset used in
    `Higher-Order Explanations of Graph Neural Networks via Relevant Walks <https://arxiv.org/abs/2006.03589>`_.
    The first class in :class:`~BA_LRP` is Barabási–Albert(BA) graph which connects a new node :math:`\mathcal{V}` from
    current graph :math:`\mathcal{G}`.

    .. math:: p(\mathcal{V}) = \frac{Degree(\mathcal{V})}{\sum_{\mathcal{V}' \in \mathcal{G}} Degree(\mathcal{V}')}

    The second class in :class:`~BA_LRP` has a slightly higher growth model and nodes are selected
    without replacement with the inverse preferential attachment model.

    .. math:: p(\mathcal{V}) = \frac{Degree(\mathcal{V})^{-1}}{\sum_{\mathcal{V}' \in \mathcal{G}} Degree(\mathcal{V}')^{-1}}

    Args:
        root (:obj:`str`): Root data directory to save datasets
        num_per_class (:obj:`int`): The number of the graphs for each class.
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    .. note:: :class:`~BA_LRP` will automatically generate the dataset
      if the dataset file is not existed in the root directory.

    Example:
        >>> dataset = BA_LRP(root='./datasets')
        >>> loader = Dataloader(dataset, batch_size=32)
        >>> data = next(iter(loader))
        # Batch(batch=[640], edge_index=[2, 1344], x=[640, 1], y=[32, 1])

    Where the attributes of data indices:

    - :obj:`batch`: The assignment vector mapping each node to its graph index
    - :obj:`x`: The node features
    - :obj:`edge_index`: The edge matrix
    - :obj:`y`: The graph label

    """
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/ba_lrp.pt')

    def __init__(self, root, num_per_class=10000, transform=None, pre_transform=None):
        self.name = 'ba_lrp'
        self.num_per_class = num_per_class
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"raw.pt"]

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def download(self):
        url = self.url
        path = download_url(url, self.raw_dir)
        shutil.move(path, path.replace('ba_lrp.pt', 'raw.pt'))

    @staticmethod
    def gen_class1():
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[0]], dtype=torch.float))

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = prob_dist.sample().squeeze()
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)
            data.y = torch.cat([data.y, torch.tensor([[0]], dtype=torch.float)], dim=0)

        return data

    @staticmethod
    def gen_class2():
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[1]], dtype=torch.float))
        epsilon = 1e-30

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg_reciprocal = torch.stack([1 / ((data.edge_index[0] == node_idx).float().sum() + epsilon) for node_idx in range(i)], dim=0)
            sum_deg_reciprocal = deg_reciprocal.sum(dim=0, keepdim=True)
            probs = (deg_reciprocal / sum_deg_reciprocal).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = -1
            for _ in range(1 if i % 5 != 4 else 2):
                new_node_pick = prob_dist.sample().squeeze()
                while new_node_pick == node_pick:
                    new_node_pick = prob_dist.sample().squeeze()
                node_pick = new_node_pick
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)
                data.y = torch.cat([data.y, torch.tensor([[1]], dtype=torch.float)], dim=0)
        return data

    def process(self):
        if files_exist(self.raw_paths):
            shutil.copyfile(self.raw_paths[0], self.processed_paths[0])
            return

        data_list = []
        for i in range(self.num_per_class):
            data_list.append(self.gen_class1())
            data_list.append(self.gen_class2())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # lrp_dataset = BA_LRP(root='.', num_per_class=10000)
    syn_dataset = SynGraphDataset(root='.', name='BA_Community')
