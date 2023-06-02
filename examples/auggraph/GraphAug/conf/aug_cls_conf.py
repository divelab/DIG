# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import copy
from dig.auggraph.method.GraphAug.constants import *

base_aug_cls_conf = {
    MODEL_NAME: CLSModelType.GCN,
    NUM_LAYERS: 4,
    HIDDEN_UNITS: 128,
    DROPOUT: 0.5,
    POOL_TYPE: PoolType.MEAN,
    BATCH_SIZE: 32,
    INITIAL_LR: 0.001,
    FACTOR: 0.5,
    PATIENCE: 5,
    MIN_LR: 0.0000001,
    MAX_NUM_EPOCHS: 100,
    AUG_MODEL_PATH: None
}

aug_cls_conf = {}

aug_cls_conf[DatasetName.NCI1] = copy.deepcopy(base_aug_cls_conf)
aug_cls_conf[DatasetName.NCI1][NUM_LAYERS] = 3
aug_cls_conf[DatasetName.NCI1][PATIENCE] = 100

aug_cls_conf[DatasetName.COLLAB] = copy.deepcopy(base_aug_cls_conf)

aug_cls_conf[DatasetName.MUTAG] = copy.deepcopy(base_aug_cls_conf)
aug_cls_conf[DatasetName.MUTAG][BATCH_SIZE] = 16
aug_cls_conf[DatasetName.MUTAG][PATIENCE] = 100

aug_cls_conf[DatasetName.PROTEINS] = copy.deepcopy(base_aug_cls_conf)
aug_cls_conf[DatasetName.PROTEINS][NUM_LAYERS] = 3

aug_cls_conf[DatasetName.IMDB_BINARY] = copy.deepcopy(base_aug_cls_conf)
aug_cls_conf[DatasetName.IMDB_BINARY][HIDDEN_UNITS] = 64

aug_cls_conf[DatasetName.NCI109] = copy.deepcopy(base_aug_cls_conf)
aug_cls_conf[DatasetName.NCI109][PATIENCE] = 100
