import random
import torch
import numpy as np
from torch_geometric.data import Batch, Data


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
    def __init__(self, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5):
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean = mask_mean
        self.mask_std = mask_std
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()
        mask = torch.zeros(node_num)

        if self.mode == 'whole':
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.random.normal(loc=self.mask_mean, scale=self.mask_std, 
                                                        size=(mask_num, feat_dim)), dtype=torch.float32)
            mask[idx_mask] = 1

        elif self.mode == 'partial':
            for i in range(node_num):
                for j in range(feat_dim):
                    if random.random() < self.mask_ratio:
                        x[i][j] = torch.tensor(np.random.normal(loc=self.mask_mean, 
                                                                scale=self.mask_std), dtype=torch.float32)
                        mask[i][j] = 1

        elif self.mode == 'onehot':
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32)
            mask[idx_mask] = 1

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(mode))

        return Data(x=x, edge_index=data.edge_index, mask=mask)

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

