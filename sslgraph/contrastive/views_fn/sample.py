import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Batch, Data


def uniform_sample(ratio=0.1):
    '''
    Args:
        ratio: Percentage of nodes to drop.
    '''
    def do_trans(data):
        
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()

        drop_num = int(node_num * ratio)
        idx_drop = np.random.choice(node_num, drop_num, replace=False)
        idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
        adj = to_dense_adj(data.edge_index)[0]
        adj = adj[idx_nondrop, :][:, idx_nondrop]
        
        return Data(x=data.x[idx_nondrop], edge_index=dense_to_sparse(adj)[0])
    
    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    batch tensor with shape [num_nodes];
                    x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    edge_index tensor with shape [2, num_edges].

        Returns:
            x tensor with shape [num_nondrop_nodes, num_node_features];
            edge_index tensor with shape [2, num_nondrop_edges];
            batch tensor with shape [num_nondrop_nodes].
        '''
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn


def RW_sample(ratio=0.1, add_self_loop=True):
    '''
    Args:
        ratio: Percentage of nodes to sample from the graph.
        add_self_loop (bool): Set True if add self-loop in edge_index.
    '''
    def do_trans(data):
        node_num, _ = data.x.size()
        sub_num = int(node_num * ratio)

        if add_self_loop:
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
        adj = to_dense_adj(edge_index)[0]
        adj = adj[idx_sampled, :][:, idx_sampled]

        return Data(x=data.x[idx_sampled], edge_index=dense_to_sparse(adj)[0])

    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    batch tensor with shape [num_nodes];
                    x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    edge_index tensor with shape [2, num_edges].

        Returns:
            x tensor with shape [num_sampled_nodes, num_node_features];
            edge_index tensor with shape [2, num_sampled_edges];
            batch tensor with shape [num_sampled_nodes].
        '''
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn

