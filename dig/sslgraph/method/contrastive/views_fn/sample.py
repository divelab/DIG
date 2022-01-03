import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.data import Batch, Data


class UniformSample():
    r"""Uniformly node dropping on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optinal): Ratio of nodes to be dropped. (default: :obj:`0.1`)
    """
    def __init__(self, ratio=0.1):
        self.ratio = ratio
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        
        node_num, _ = data.x.size()
        device = data.x.device
        _, edge_num = data.edge_index.size()
        
        keep_num = int(node_num * (1-self.ratio))
        idx_nondrop = torch.randperm(node_num, device=device)[:keep_num]
        mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_nondrop, 1.0).bool()
        
        edge_index, _ = subgraph(mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
        return Data(x=data.x[mask_nondrop], edge_index=edge_index)
    
    def views_fn(self, data):
        r"""Method to be called when :class:`UniformSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)



class RWSample():
    """Subgraph sampling based on random walk on the given graph or batched graphs.
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optional): Percentage of nodes to sample from the graph.
            (default: :obj:`0.1`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`False`)
    """
    def __init__(self, ratio=0.1, add_self_loop=False):
        self.ratio = ratio
        self.add_self_loop = add_self_loop
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        device = data.x.device
        node_num, _ = data.x.size()
        sub_num = int(node_num * self.ratio)

        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)], device=device).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index

        idx_sub = [torch.randint(node_num, size=(1,), device=device)[0]]
        idx_neigh = set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = list(idx_neigh)[torch.randperm(len(idx_neigh), device=device)[0]]
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_sub = torch.LongTensor(idx_sub, device=device)
        mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_sub, 1.0).bool()
        edge_index, _ = subgraph(mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
        return Data(x=data.x[mask_nondrop], edge_index=edge_index)

    def views_fn(self, data):
        r"""Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)