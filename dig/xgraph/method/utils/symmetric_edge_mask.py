import torch

def symmetric_edge_mask_indirect_graph(edge_index: 'torch.Tensor', edge_mask: 'torch.Tensor') -> 'torch.Tensor':
    """ makes the given edge_mask symmetric, provided the given graph is indirect one. 
    
    Args:
        edge_index (torch.Tensor): edges of the target graph
        edge_mask (torch.Tensor): edge mask provided by the explainer for the graph
    """
    # checkout if graph is indirect one
    def _is_indirect() -> bool:
        with torch.no_grad():
            edge_index_src = edge_index.detach().unsqueeze(2)   # shape: 2 N 1
            edge_index_rev = edge_index_src[[1, 0]].transpose(1, 2)   # 2 1 N

            eq = edge_index_src - edge_index_rev == 0    # 2 N N
            rev_exist = (eq[0] * eq[1]).sum(1) # N
            return torch.all(rev_exist > 0).item()

    if _is_indirect():
        edge_mask = edge_mask.to(edge_index.device)

        num_nodes = edge_index.unique().numel()
        edge_mask_asym = torch.sparse_coo_tensor(edge_index, 
                edge_mask, (num_nodes, num_nodes)).to_dense()
        edge_mask_sym = (edge_mask_asym + edge_mask_asym.T) / 2
        edge_mask = edge_mask_sym[edge_index[0], edge_index[1]]

    return edge_mask
