"""
Modified from the https://github.com/rusty1s/pyg_autoscale/tree/master/torch_geometric_autoscale/models
"""

from .base import ScalableGNN
from .gcn import GCN
from .gcn2 import GCN2
from .pna import PNA
from .pna_jk import PNA_JK

__all__ = [
    'ScalableGNN',
    'GCN',
    'GCN2',
    'PNA',
    'PNA_JK',
]
