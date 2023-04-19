# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

from enum import Enum

class FuseType(Enum):
    ABS_DIFF = 'abs_diff'
    CONCAT = 'concat'
    COSINE = 'cosine'
    ADD = 'add'
    MULTIPLY = 'multiply'

class AugType(Enum):
    NODE_FM = 'node_fm'
    NODE_DROP = 'node_drop'
    EDGE_Per = 'edge_per'

class BaselineType(Enum):
    EXP = 'exp'
    MEAN = 'mean'

class DatasetName(Enum):
    NCI1 = 'NCI1'
    COLLAB = 'COLLAB'
    MUTAG = 'MUTAG'
    PROTEINS = 'PROTEINS'
    IMDB_BINARY = 'IMDB-BINARY'
    NCI109 = 'NCI109'
    AIDS = 'AIDS'

class PoolType(Enum):
    SUM = 'sum'
    MEAN = 'mean'
    MAX = 'max'

class RewardGenModelType(Enum):
    GMNET = 'gmnet'
    GENET = 'genet'

class RnnInputType(Enum):
    VIRTUAL = 'virtual'
    ONE_HOT = 'one-hot'

class NodeUpdateType(Enum):
    MLP = 'mlp'
    RESIDUAL = 'residual'
    GRU = 'gru'

class ReduceType(Enum):
    ADD = 'add'

class ConvType(Enum):
    GEMB = 'gemb'
    GIN = 'gin'

class MaskType(Enum):
    ZERO = 'zero'
    GAUSSIAN = 'gaussian'

class CLSModelType(Enum):
    GIN = 'gin'
    GCN = 'gcn'
