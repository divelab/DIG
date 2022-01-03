from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, Sequential


class GRACE(Contrastive):
    r"""
    Contrastive learning method proposed in the paper `Deep Graph Contrastive Representation 
    Learning <https://arxiv.org/abs/2006.04131>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_grace.ipynb>`_ for
    an example of usage.
    
    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GRACE`.
        
    Args:
        dim (int): The embedding dimension.
        dropE_rate_1, dropE_rate_2 (float): The ratio of the edge dropping augmentation for 
            view 1. A number between [0,1).
        maskN_rate_1, maskN_rate_2 (float): The ratio of the node masking augmentation for
            view 2. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, dim, dropE_rate_1, dropE_rate_2, maskN_rate_1, maskN_rate_2, 
                 **kwargs):

        view_fn_1 = Sequential([EdgePerturbation(drop=True, ratio=dropE_rate_1),
                                NodeAttrMask(mask_ratio=maskN_rate_1)])
        view_fn_2 = Sequential([EdgePerturbation(drop=True, ratio=dropE_rate_2),
                                NodeAttrMask(mask_ratio=maskN_rate_2)])
        views_fn = [view_fn_1, view_fn_2]
        
        super(GRACE, self).__init__(objective='NCE',
                                    views_fn=views_fn,
                                    graph_level=False,
                                    node_level=True,
                                    z_n_dim=dim,
                                    proj_n='MLP',
                                    **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GRACE removes projection heads after pre-training
        for enc, proj in super().train(encoders, data_loader, 
                                       optimizer, epochs, per_epoch_out):
            yield enc
