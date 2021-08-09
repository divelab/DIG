import itertools
import torch
import torch.nn.functional as F


def NCE_loss(zs=None, zs_n=None, batch=None, sigma=None, **kwargs):
    '''The InfoNCE (NT-XENT) loss in contrastive learning.
    
    Args:
        zs (list, optipnal): List of tensors of shape [batch_size, z_dim].
        zs_n (list, optional): List of tensors of shape [nodes, z_dim].
        batch (Tensor, optional): Required when both :obj:`zs` and :obj:`zs_n` are given.
        sigma (ndarray, optional): A 2D-array of shape [:obj:`n_views`, :obj:`n_views`] with boolean 
            values, indicating contrast between which two views are computed. Only required 
            when number of views is greater than 2. If :obj:`sigma[i][j]` = :obj:`True`, 
            infoNCE between :math:`view_i` and :math:`view_j` will be computed.
        tau (int, optional): The temperature used in NT-XENT.

    :rtype: :class:`Tensor`
    '''
    assert zs is not None or zs_n is not None
    
    if 'tau' in kwargs:
        tau = kwargs['tau']
    else:
        tau = 0.5
    
    if 'norm' in kwargs:
        norm = kwargs['norm']
    else:
        norm = True
        
    if zs_n is not None:
        if zs is None:
            # InfoNCE in GRACE
            assert len(zs_n)==2
            return (infoNCE_local_intra_node(zs_n[0], zs_n[1], tau, norm)+
                    infoNCE_local_intra_node(zs_n[1], zs_n[0], tau, norm))*0.5
        else:
            assert len(zs_n)==len(zs)
            assert batch is not None
            
            if len(zs)==1:
                return infoNCE_local_global(zs[0], zs_n[0], batch, tau, norm)
            elif len(zs)==2:
                return (infoNCE_local_global(zs[0], zs_n[1], batch, tau, norm)+
                        infoNCE_local_global(zs[1], zs_n[0], batch, tau, norm))
            else:
                assert len(zs)==len(sigma)
                loss = 0
                for (i, j) in itertools.combinations(range(len(zs)), 2):
                    if sigma[i][j]:
                        loss += (infoNCE_local_global(zs[i], zs_n[j], batch, tau, norm)+
                                 infoNCE_local_global(zs[j], zs_n[i], batch, tau, norm))
                return loss
    
    if len(zs)==2:
        return NT_Xent(zs[0], zs[1], tau, norm)
    elif len(zs)>2:
        assert len(zs)==len(sigma)
        loss = 0
        for (i, j) in itertools.combinations(range(len(zs)), 2):
            if sigma[i][j]:
                loss += NT_Xent(zs[i], zs[j], tau, norm)
        return loss

    
def infoNCE_local_intra_node(z1_n, z2_n, tau=0.5, norm=True):
    '''
    Args:
        z1_n: Tensor of shape [n_nodes, z_dim].
        z2_n: Tensor of shape [n_nodes, z_dim].
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    if norm:
        z1_n = F.normalize(z1_n)
        z2_n = F.normalize(z2_n)
    exp = lambda x: torch.exp(x / tau)
    refl_sim = exp(torch.mm(z1_n, z1_n.t()))
    between_sim = exp(torch.mm(z1_n, z2_n.t()))
    
    pos_sim = between_sim.diag()
    intra_sim = refl_sim.sum(1) - refl_sim.diag()
    inter_pos_sim = between_sim.sum(1)
    
    loss = pos_sim / (intra_sim + inter_pos_sim)
    loss = -torch.log(loss).mean()

    return loss
    
    
                
def infoNCE_local_global(z_n, z_g, batch, tau=0.5, norm=True):
    '''
    Args:
        z_n: Tensor of shape [n_nodes, z_dim].
        z_g: Tensor of shape [n_graphs, z_dim].
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    # Not yet used in existing methods, to be implemented.
    loss = 0

    return loss



def NT_Xent(z1, z2, tau=0.5, norm=True):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    
    batch_size, _ = z1.size()
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
    
    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
        
    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss
