import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, dropout_adj, degree, add_remaining_self_loops
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Batch, Data
from dig.sslgraph.utils.adaptive import degree_drop_weights, pr_drop_weights, evc_drop_weights, drop_edge_weighted

class EdgePerturbation():
    '''Edge perturbation on the given graph or batched graphs. Class objects callable via 
    method :meth:`views_fn`.
    
    Args:
        add (bool, optional): Set :obj:`True` if randomly add edges in a given graph.
            (default: :obj:`True`)
        drop (bool, optional): Set :obj:`True` if randomly drop edges in a given graph.
            (default: :obj:`False`)
        ratio (float, optional): Percentage of edges to add or drop. (default: :obj:`0.1`)
    '''
    def __init__(self, add=True, drop=False, ratio=0.1):
        self.add = add
        self.drop = drop
        self.ratio = ratio
        
    def __call__(self, data):
        return self.views_fn(data)
        
    def do_trans(self, data):
        node_num, _ = data.x.size()
        device = data.x.device
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * self.ratio)
        idx_add = torch.tensor([], device=device).reshape(2, -1).long()

        if self.drop:
            idx_remain = dropout_adj(data.edge_index, p=self.ratio)[0]

        if self.add:
            idx_add = torch.randint(node_num, (2, perturb_num), device=device)

        new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        return Data(x=data.x, edge_index=new_edge_index)

    def views_fn(self, data):
        r"""Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)

class AdaEdgePerturbation():
    '''Proabilistic edge perturbation on the given graph or batched graphs. Perturbations are adaptive i.e. critical edges have lower probability of being removed.
    Adaptations based on the paper `Graph Contrastive Learning with Adaptive Augmentation <https://arxiv.org/abs/2010.14945>`
    Refer to `source implementation <https://github.com/CRIPAC-DIG/GCA>` for more details.
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        centrality_measure (str): Method for computing edge centrality. Set `degree` for degree centrality,
        `pr` for PageRank centrality and `evc` for eigen-vector centrality
        prob (float): Probability factor used for calculating edge-drop probability
        threshold (float): Upper-bound probability for dropping any edge, defaults to 0.7
    '''
    def __init__(self, centrality_measure : str, prob : float, threshold : float):
        self.centrality_measure = centrality_measure
        self.prob = prob
        self.threshold = threshold
    def __call__(self, data):
        return self.views_fn(data)
    def do_trans(self, data):
        if self.centrality_measure not in ['degree', 'pr', 'evc']:
            raise ValueError("Invalid drop scheme {}".format(self.centrality_measure))
        drop_weights = self._get_edge_weights(data)
        new_edge_index = drop_edge_weighted(data.edge_index, drop_weights, p=self.prob, threshold=self.threshold)
        return Data(x=data.x, edge_index=new_edge_index)
        
    def _get_edge_weights(self, data):
        # Add self loops if the degree returns for only a subset of nodes
        if degree(data.edge_index[1]).size(0) != data.x.size(0):
            edge_index_ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.size(0))[0]
        else:
            edge_index_ = data.edge_index
        if self.centrality_measure == 'degree':
            drop_weights = degree_drop_weights(edge_index_)
        elif self.centrality_measure == 'pr':
            drop_weights = pr_drop_weights(edge_index_, aggr='sink', k=200)
        elif self.centrality_measure == 'evc':
            drop_weights = evc_drop_weights(data)
        return drop_weights
    def views_fn(self, data):
        r"""Method to be called when :class:`AdaEdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)

class Diffusion():
    '''Diffusion on the given graph or batched graphs, used in 
    `MVGRL <https://arxiv.org/pdf/2006.05582v1.pdf>`_. Class objects callable via 
    method :meth:`views_fn`.
    
    Args:
        mode (string, optional): Diffusion instantiation mode with two options:
            :obj:`"ppr"`: Personalized PageRank; :obj:`"heat"`: heat kernel.
            (default: :obj:`"ppr"`)
        alpha (float, optinal): Teleport probability in a random walk. (default: :obj:`0.2`)
        t (float, optinal): Diffusion time. (default: :obj:`5`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`True`)
    '''
    def __init__(self, mode='ppr', alpha=0.2, t=5, add_self_loop=True):
        self.mode = mode
        self.alpha = alpha
        self.t = t
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        node_num, _ = data.x.size()
        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)], device=data.x.device).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()
        
        orig_adj = to_dense_adj(edge_index)[0]
        orig_adj = torch.where(orig_adj>1, torch.ones_like(orig_adj), orig_adj)
        d = torch.diag(torch.sum(orig_adj, 1))

        if self.mode == 'ppr':
            dinv = torch.inverse(torch.sqrt(d))
            at = torch.matmul(torch.matmul(dinv, orig_adj), dinv)
            diff_adj = self.alpha * torch.inverse((torch.eye(orig_adj.shape[0]) - (1 - self.alpha) * at))

        elif self.mode == 'heat':
            diff_adj = torch.exp(self.t * (torch.matmul(orig_adj, torch.inverse(d)) - 1))

        else:
            raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")
            
        edge_ind, edge_attr = dense_to_sparse(diff_adj)

        return Data(x=data.x, edge_index=edge_ind, edge_attr=edge_attr)
    

    def views_fn(self, data):
        r"""Method to be called when :class:`Diffusion` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)



class DiffusionWithSample():
    '''Diffusion with node sampling on the given graph for node-level datasets, used in 
    `MVGRL <https://arxiv.org/pdf/2006.05582v1.pdf>`_.  Class objects callable via method 
    :meth:`views_fn`.
    
    Args:
        sample_size (int, optional): Number of nodes in the sampled subgraoh from a large graph.
            (default: :obj:`2000`)
        batch_size (int, optional): Number of subgraphs to sample. (default: :obj:`4`)
        mode (string, optional): Diffusion instantiation mode with two options:
            :obj:`"ppr"`: Personalized PageRank; :obj:`"heat"`: heat kernel; 
            (default: :obj:`"ppr"`)
        alpha (float, optinal): Teleport probability in a random walk. (default: :obj:`0.2`)
        t (float, optinal): Diffusion time. (default: :obj:`5`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`True`)
    '''
    def __init__(self, sample_size=2000, batch_size=4, mode='ppr', 
                 alpha=0.2, t=5, epsilon=False, add_self_loop=True):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.mode = mode
        self.alpha = alpha
        self.t = t
        self.epsilon = epsilon
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        return self.views_fn(data)
    
    def views_fn(self, data):
        r"""Method to be called when :class:`DiffusionWithSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        node_num, _ = data.x.size()
        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)], device=data.x.device).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()
        
        orig_adj = to_dense_adj(edge_index)[0]
        orig_adj = torch.where(orig_adj>1, torch.ones_like(orig_adj), orig_adj)
        d = torch.diag(torch.sum(orig_adj, 1))

        if self.mode == 'ppr':
            dinv = torch.inverse(torch.sqrt(d))
            at = torch.matmul(torch.matmul(dinv, orig_adj), dinv)
            diff_adj = self.alpha * torch.inverse((torch.eye(orig_adj.shape[0]) - (1 - self.alpha) * at))

        elif self.mode == 'heat':
            diff_adj = torch.exp(self.t * (torch.matmul(orig_adj, torch.inverse(d)) - 1))

        else:
            raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")

        if self.epsilon:
            epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
            avg_degree = torch.sum(orig_adj) / orig_adj.shape[0]
            ep = epsilons[torch.argmin([abs(avg_degree - torch.sum(diff_adj >= e) / diff_adj.shape[0]) for e in epsilons])]

            diff_adj[diff_adj < ep] = 0.0
            scaler = MinMaxScaler()
            scaler.fit(diff_adj)
            diff_adj = torch.tensor(scaler.transform(diff_adj), device=data.x.device)

        dlist_orig_x = []
        dlist_diff_x = []
        drop_num = node_num - self.sample_size
        for b in range(self.batch_size):
            idx_drop = torch.randperm(node_num, device=data.x.device)[:drop_num]
            idx_nondrop = [n for n in range(node_num) if not n in idx_drop]

            sample_orig_adj = orig_adj.clone()
            sample_orig_adj = sample_orig_adj[idx_nondrop, :][:, idx_nondrop]

            sample_diff_adj = diff_adj.clone()
            sample_diff_adj = sample_diff_adj[idx_nondrop, :][:, idx_nondrop]

            sample_orig_x = data.x[idx_nondrop]
            
            edge_ind, edge_attr = dense_to_sparse(sample_diff_adj)

            dlist_orig_x.append(Data(x=sample_orig_x, 
                                     edge_index=dense_to_sparse(sample_orig_adj)[0]))
            dlist_diff_x.append(Data(x=sample_orig_x, 
                                     edge_index=edge_ind, 
                                     edge_attr=edge_attr))

        return Batch.from_data_list(dlist_orig_x), Batch.from_data_list(dlist_diff_x)
