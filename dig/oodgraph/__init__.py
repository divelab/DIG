r"""
This module includes 8 GOOD datasets.

- Graph prediction datasets: GOOD-HIV, GOOD-PCBA, GOOD-ZINC, GOOD-CMNIST, GOOD-Motif.
- Node prediction datasets: GOOD-Cora, GOOD-Arxiv, GOOD-CBAS.
"""

from .good_hiv import GOODHIV
from .good_arxiv import GOODArxiv
from .good_pcba import GOODPCBA
from .good_cmnist import GOODCMNIST
from .good_cora import GOODCora
from .good_cbas import GOODCBAS
from .good_motif import GOODMotif
from .good_zinc import GOODZINC

__all__ = [
    'GOODCBAS',
    'GOODZINC',
    'GOODHIV',
    'GOODCMNIST',
    'GOODArxiv',
    'GOODPCBA',
    'GOODMotif',
    'GOODCora'
]

