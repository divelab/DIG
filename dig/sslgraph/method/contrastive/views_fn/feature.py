import random
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from typing import Dict
from torch_geometric.utils import to_undirected, degree
from dig.sslgraph.utils.pgrace import eigenvector_centrality, feature_drop_weights_dense, feature_drop_weights, compute_pr, drop_feature_weighted_2


class NodeAttrMask():
    '''Node attribute masking on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    '''
    def __init__(self, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5, return_mask=False):
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.return_mask = return_mask
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()

        if self.mode == 'whole':
            mask = torch.zeros(node_num)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = torch.randperm(x.size(0), device=x.device)[:mask_num]
            if self.mask_std > 0:
                x[idx_mask] = torch.empty((mask_num, feat_dim), dtype=torch.float32, 
                    device=x.device).normal_(mean=self.mask_mean,std=self.mask_std)
            else:
                x[idx_mask] = self.mask_mean

            mask[idx_mask] = 1

        elif self.mode == 'partial':
            mask = torch.zeros((node_num, feat_dim))
            for i in range(node_num):
                for j in range(feat_dim):
                    if random.random() < self.mask_ratio:
                        x[i][j] = torch.tensor(np.random.normal(loc=self.mask_mean, 
                            scale=self.mask_std), dtype=torch.float32, device=x.device)
                        mask[i][j] = 1

        elif self.mode == 'onehot':
            mask = torch.zeros(node_num)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = torch.randperm(x.size(0), device=x.device)[:mask_num]
            x[idx_mask] = torch.eye(feat_dim, dtype=torch.float32, device=x.device
                )[torch.randint(0, feat_dim, size=(mask_num), device=x.device)]
            mask[idx_mask] = 1

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(mode))

        if self.return_mask:
            return Data(x=x, edge_index=data.edge_index, mask=mask)
        else:
            return Data(x=x, edge_index=data.edge_index)

    def views_fn(self, data):
        r"""Method to be called when :class:`NodeAttrMask` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)

class GCANodeAttrMask():
    def __init__(self, centrality_measure: str, prob: float = 0.1, threshold: float = 0.7, dense: bool = False, return_mask: bool = False):
        self.centrality_measure = centrality_measure
        self.dense = dense
        self.return_mask = return_mask
        self.prob = prob
        self.threshold = threshold
    
    def __call__(self, data):
        return self.views_fn(data)

    
    def _get_node_centrality(self, data : Data):
        if self.centrality_measure == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            node_c = node_deg
        elif self.centrality_measure == 'pr':
            node_pr = compute_pr(data.edge_index)
            node_c = node_pr
        elif self.centrality_measure == 'evc':
            node_evc = eigenvector_centrality(data)
            node_c = node_evc
        else:
            # Don't allow masking if centrality measure is not specified
            # GCA official implementation uses a full-one mask, but we mandate the user to remove NodePerturbation from the view_fn
            raise Exception("Centrality measure option '{}' is not available!".format(self.centrality_measure))
        return node_c

    def do_trans(self, data : Data):
        # TODO: determine if to(device) call is needed
        x = data.x.detach().clone()
        node_c = self._get_node_centrality(data).to(x.device)
        if self.dense:
            feature_weights = feature_drop_weights_dense(data.x, node_c).to(x.device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c).to(x.device)
        x = drop_feature_weighted_2(x, feature_weights, p = self.prob, threshold = self.threshold).to(x.device)
        if self.return_mask:
            return Data(x=x, edge_index=data.edge_index, mask=feature_weights).to(x.device)
        else:
            return Data(x=x, edge_index=data.edge_index).to(x.device)
    
    
    
    def views_fn(self, data):
        r"""Method to be called when :class:`NodeAttrMask` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
