import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def JSE_loss(zs, zs_n=None, batch=None, sigma=None, neg_by_crpt=False, **kwargs):
    '''The Jensen-Shannon Estimator of Mutual Information used in contrastive learning. The
    implementation follows the paper `Learning deep representations by mutual information 
    estimation and maximization <https://arxiv.org/abs/1808.06670>`_.
    
    .. note::
        The JSE loss implementation can produce negative values because a :obj:`-2log2` shift is 
        added to the computation of JSE, for the sake of consistency with other f-convergence 
        losses.
    
    Args:
        zs (list, optional): List of tensors of shape [batch_size, z_dim].
        zs_n (list, optional): List of tensors of shape [nodes, z_dim].
        batch (Tensor, optional): Required when both :obj:`zs` and :obj:`zs_n` are given.
        sigma (ndarray, optional): A 2D-array of shape [n_views, n_views] with boolean values, 
            indicating contrast between which two views are computed. Only required 
            when number of views is greater than 2. If :obj:`sigma[i][j]` = :obj:`True`, 
            JSE between :math:`view_i` and :math:`view_j` will be computed.
        neg_by_crpt (bool, optional): The mode to obtain negative samples in JSE. If True, 
            obtain negative samples by performing corruption. Otherwise, consider pairs of
            different graph samples as negative pairs.

    :rtype: :class:`Tensor`
    '''
    if zs_n is not None:
        assert len(zs_n) == len(zs)
        assert batch is not None
        
        jse = (JSE_local_global_negative_paired 
               if neg_by_crpt else JSE_local_global)
        
        if len(zs) == 1:
            return jse(zs[0], zs_n[0], batch)
        elif len(zs) == 2:
            return (jse(zs[0], zs_n[1], batch) +
                    jse(zs[1], zs_n[0], batch))
        else:
            assert len(zs) == len(sigma)
            loss = 0
            for (i, j) in itertools.combinations(range(len(zs)), 2):
                if sigma[i][j]:
                    loss += (jse(zs[i], zs_n[j], batch) +
                             jse(zs[j], zs_n[i], batch))
            return loss

    else:
        jse = JSE_global_global
        if len(zs) == 2:
            return jse(zs[0], zs[1])
        elif len(zs) > 2:
            assert len(zs) == len(sigma)
            loss = 0
            for (i, j) in itertools.combinations(range(len(zs)), 2):
                if sigma[i][j]:
                    loss += jse(zs[i], zs[j])
            return loss


def JSE_local_global_negative_paired(z_g, z_n, batch):
    '''
    Args:
        z_g: of size [2*n_batch, dim]
        z_n: of size [2*n_batch*nodes_per_batch, dim]
    '''
    device = z_g.device
    num_graphs = int(z_g.shape[0]/2)  # 4
    num_nodes = int(z_n.shape[0]/2) # 4*2000
    z_g, _ = torch.split(z_g, num_graphs)
    z_n, z_n_crpt = torch.split(z_n, num_nodes)

    num_sample_nodes = int(num_nodes / num_graphs)
    z_n = torch.split(z_n, num_sample_nodes)
    z_n_crpt = torch.split(z_n_crpt, num_sample_nodes)

    d_pos = torch.cat([torch.matmul(z_g[i], z_n[i].t()) for i in range(num_graphs)])  # [1, 8000]
    d_neg = torch.cat([torch.matmul(z_g[i], z_n_crpt[i].t()) for i in range(num_graphs)])  # [1, 8000]
        
    logit = torch.unsqueeze(torch.cat((d_pos, d_neg)), 0)  # [1, 16000]
    lb_pos = torch.ones((1, num_nodes)).to(device)  # [1, 8000]
    lb_neg = torch.zeros((1, num_nodes)).to(device)  # [1, 8000]
    lb = torch.cat((lb_pos, lb_neg), 1)

    b_xent = nn.BCEWithLogitsLoss()
    loss = b_xent(logit, lb) * 0.5 # following mvgrl-node
    return loss
    

def JSE_local_global(z_g, z_n, batch):
    '''
    Args:
        z_g: Tensor of shape [n_graphs, z_dim].
        z_n: Tensor of shape [n_nodes, z_dim].
        batch: Tensor of shape [n_graphs].
    '''
    device = z_g.device
    num_graphs = z_g.shape[0]
    num_nodes = z_n.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    d_prime = torch.matmul(z_n, z_g.t())

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def JSE_global_global(z1, z2):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim].
    '''
    device = z1.device
    num_graphs = z1.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).to(device)
    neg_mask = torch.ones((num_graphs, num_graphs)).to(device)
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    d_prime = torch.matmul(z1, z2.t())

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def get_expectation(masked_d_prime, positive=True):
    '''
    Args:
        masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global,
                        tensor of shape [n_nodes, n_graphs] for local_global.
        positive (bool): Set True if the d_prime is masked for positive pairs,
                        set False for negative pairs.
    '''
    log_2 = np.log(2.)
    if positive:
        score = log_2 - F.softplus(-masked_d_prime)
    else:
        score = F.softplus(-masked_d_prime) + masked_d_prime - log_2
    return score

