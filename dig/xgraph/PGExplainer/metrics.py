import torch
import numpy as np
from torch_geometric.data import Data, Batch
from typing import Optional


def calculate_selected_nodes(data, edge_mask, top_k, node_idx=None):
    threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
    hard_mask = (edge_mask > threshold).cpu()
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    edge_index = data.edge_index.cpu().numpy()
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
    selected_nodes = list(set(selected_nodes))
    if node_idx is not None:
        selected_nodes.append(node_idx)
    return selected_nodes


def top_k_fidelity(data: Data, edge_mask: np.array, top_k: int,
                   gnnNets: torch.nn.Module, label: int,
                   target_id: int = -1, node_idx: Optional[int]=None,
                   undirected=True):
    """ return the fidelity score of the subgraph with top_k score edges  """
    if undirected:
        top_k = 2 * top_k
    all_nodes = np.arange(data.x.shape[0]).tolist()
    selected_nodes = calculate_selected_nodes(data, edge_mask, top_k, node_idx)
    score = gnn_score(all_nodes, data, gnnNets, label, target_id, node_idx=node_idx,
                      subgraph_building_method='zero_filling')
    unimportant_nodes = [node for node in all_nodes if node not in selected_nodes]
    score_mask_important = gnn_score(unimportant_nodes, data, gnnNets, label, target_id, node_idx=node_idx,
                                     subgraph_building_method='zero_filling')
    return score - score_mask_important


def top_k_sparsity(data: Data, edge_mask: np.array, top_k: int, undirected=True):
    """ return the size ratio of the subgraph with top_k score edges"""
    if undirected:
        top_k = 2 * top_k
    selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
    return 1 - len(selected_nodes) / data.x.shape[0]


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError


def graph_build_zero_filling(X, edge_index, node_mask: np.array):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: np.array):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index


def gnn_score(coalition: list, data: Data, gnnNets, label: int,
              target_id: int = -1, node_idx=None, subgraph_building_method='zero_filling') -> torch.Tensor:
    """ the prob of subgraph with selected nodes for required label and target node """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    logits, probs, _ = gnnNets(mask_data)

    # get the score of predicted class for graph or specific node idx
    node_idx = 0 if node_idx is None else node_idx
    if target_id == -1:
        score = probs[node_idx, label].item()
    else:
        score = probs[node_idx, target_id, label].item()
    return score
