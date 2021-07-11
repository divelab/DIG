from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
    UniformSample, RWSample, RandomView


class GraphCL(Contrastive):
    r"""    
    Contrastive learning method proposed in the paper `Graph Contrastive Learning with 
    Augmentations <https://arxiv.org/abs/2010.13902>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_graphcl.ipynb>`_ for
    an example of usage.

    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GraphCL`.
    
    Args:
        dim (int): The embedding dimension.
        aug1 (sting, optinal): Types of augmentation for the first view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug2 (sting, optinal): Types of augmentation for the second view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug_ratio (float, optional): The ratio of augmentations. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, dim, aug_1=None, aug_2=None, aug_ratio=0.2, **kwargs):
        
        views_fn = []
        
        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(UniformSample(ratio=aug_ratio))
            elif aug == 'permE':
                views_fn.append(EdgePerturbation(ratio=aug_ratio))
            elif aug == 'subgraph':
                views_fn.append(RWSample(ratio=aug_ratio))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=aug_ratio))
            elif aug == 'random2':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random4':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random3':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio),
                              NodeAttrMask(mask_ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            else:
                raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                'maskN', 'random2', 'random3', 'random4'] or None.")
        
        super(GraphCL, self).__init__(objective='NCE',
                                      views_fn=views_fn,
                                      z_dim=dim,
                                      proj='MLP',
                                      node_level=False,
                                      **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, proj in super(GraphCL, self).train(encoders, data_loader, 
                                                    optimizer, epochs, per_epoch_out):
            yield enc
