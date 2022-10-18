# Copyright (c) 2021 Big Data and Multi-modal Computing Group, CRIPAC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .contrastive import Contrastive
from dig.sslgraph.method.contrastive.views_fn import AdaNodeAttrMask, AdaEdgePerturbation, Sequential
import torch


class pGRACE(Contrastive):
    r"""
    Adaptive augmentation methods proposed in the paper `Graph Contrastive Learning
    with Adaptive Augmentation <https://arxiv.org/abs/2010.14945>`_. You can refer to `the original implementation 
    <https://github.com/CRIPAC-DIG/GCA>` or `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_grace.ipynb>`_ for
    an example of usage.
    
    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`pGRACE`.
        
    Args:
        dim (int): The embedding dimension.
        proj_n_dim (int): The projection head dimension to use for computing loss
        centrality_measure (str): The metric to use for computing edge or node centrality. Supported values are `degree`, `evc` (Eigen-Vector centrality) and `pr` (PageRank centrality).
        prob_edge_1, prob_edge_2 (float): The probability factor for calculating edge-drop probability
        prob_feature_1, prob_feature_2 (float): The probability factor for calculating feature-masking probability
        tau (float, optional): The temperature parameter used for contrastive objective.
        dense (bool, optional): Whether the node features are dense continuous features. Defaults to `false`.
        p_tau (float, optional): The upper-bound probability for dropping edges or removing nodes.
        **kwargs (optional): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, dim: int, proj_n_dim: int, centrality_measure: str, prob_edge_1: float, prob_edge_2: float,
                prob_feature_1: float, prob_feature_2: float, tau: float = 0.1, dense:bool = False, p_tau: float = 0.7, **kwargs):
        view_fn_1 = Sequential([AdaEdgePerturbation(centrality_measure, prob=prob_edge_1, threshold=p_tau),
                                AdaNodeAttrMask(centrality_measure, prob=prob_feature_1, threshold=p_tau, dense=dense)])
        view_fn_2 = Sequential([AdaEdgePerturbation(centrality_measure, prob=prob_edge_2, threshold=p_tau),
                                AdaNodeAttrMask(centrality_measure, prob=prob_feature_2, threshold=p_tau, dense=dense)])
        views_fn = [view_fn_1, view_fn_2]

        device = kwargs['device'] if 'device' in kwargs else 0
        
        super(pGRACE, self).__init__(objective='NCE',
                                    views_fn=views_fn,
                                    graph_level=False,
                                    node_level=True,
                                    z_n_dim=dim,
                                    tau=tau,
                                    proj_n=self._proj_head,
                                    **kwargs)
        
        self.proj_n = torch.nn.Sequential(
            torch.nn.Linear(dim, proj_n_dim),
            torch.nn.ELU(),
            torch.nn.Linear(proj_n_dim, dim)
        ).to(device)
    
    def _proj_head(self, x):
        return self.proj_n(x)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GRACE removes projection heads after pre-training
        for enc, proj in super().train(encoders, data_loader, 
                                       optimizer, epochs, per_epoch_out):
            yield enc
