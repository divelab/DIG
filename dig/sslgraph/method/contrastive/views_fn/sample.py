import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
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
        _, edge_num = data.edge_index.size()

        drop_num = int(node_num * self.ratio)
        idx_drop = np.random.choice(node_num, drop_num, replace=False)
        idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
        adj = to_dense_adj(data.edge_index, max_num_nodes=node_num)[0]
        adj = adj[idx_nondrop, :][:, idx_nondrop]
        
        return Data(x=data.x[idx_nondrop], edge_index=dense_to_sparse(adj)[0])
    
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
        node_num, _ = data.x.size()
        sub_num = int(node_num * self.ratio)

        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()

        # edge_index = edge_index.numpy()
        idx_sub = [np.random.randint(node_num, size=1)[0]]
        # idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
        idx_neigh = set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            # idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))
            idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_sampled = idx_sub
        adj = to_dense_adj(edge_index, max_num_nodes=node_num)[0]
        adj = adj[idx_sampled, :][:, idx_sampled]

        return Data(x=data.x[idx_sampled], edge_index=dense_to_sparse(adj)[0])

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


