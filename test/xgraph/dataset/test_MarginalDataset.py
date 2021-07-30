import torch
import numpy as np
from torch_geometric.data import Data
from dig.xgraph.dataset import MarginalSubgraphDataset


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_x = X * node_mask.unsqueeze(1)
    return ret_x, edge_index


def test_MarginalSubgraphDataset():
    num_mask = 10
    num_nodes = 6
    x = torch.ones(num_nodes, 10)
    edge_index = torch.LongTensor([[0, 1, 1, 2, 2, 3, 4, 5, 5],
                                   [2, 2, 5, 0, 1, 5, 5, 1, 3]])
    y = torch.LongTensor([0, 1, 1, 0, 0, 1])
    data = Data(x=x, edge_index=edge_index, y=y)

    node_indices = list(range(num_nodes))
    coalition = [1, 2]
    coalition_placeholder = num_nodes
    set_include_masks = []
    set_exclude_masks = []
    for mask_idx in range(num_mask):
        subset_nodes_from = [node for node in node_indices if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.zeros(num_nodes)
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_dataloader = \
        MarginalSubgraphDataset(data=data,
                                exclude_mask=exclude_mask,
                                include_mask=include_mask,
                                subgraph_build_func=graph_build_zero_filling)

    exclude_data, include_data = marginal_dataloader[0]
    assert exclude_data.x.shape == (6, 10)
    assert exclude_data.edge_index.shape == (2, 9)

    assert include_data.x.shape == (6, 10)
    assert exclude_data.edge_index.shape == (2, 9)

    assert marginal_dataloader.X.shape == (6, 10)
    assert marginal_dataloader.edge_index.shape == (2, 9)


if __name__ == '__main__':
    test_MarginalSubgraphDataset()
