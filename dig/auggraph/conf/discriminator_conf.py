import copy
from ..constants import *

base_dis_conf = {
    BATCH_SIZE: 32,
    INITIAL_LR: 1e-4,
    FACTOR: 0.5,
    PATIENCE: 5,
    MIN_LR: 1e-7,
    MAX_NUM_EPOCHS: 200,
    PRE_TRAIN_PATH: None,
    DISCRIMINATOR_PARAMS: {
        NUM_LAYERS: 5,
        HIDDEN_UNITS: 256,
        MODEL_TYPE: GMNET,
        POOL_TYPE: SUM,
        FUSE_TYPE: ABS_DIFF
    }
}

dis_conf = {}

dis_conf[NCI1] = copy.deepcopy(base_dis_conf)

dis_conf[COLLAB] = copy.deepcopy(base_dis_conf)
dis_conf[COLLAB][BATCH_SIZE] = 8
dis_conf[COLLAB][MAX_NUM_EPOCHS] = 120

dis_conf[MUTAG] = copy.deepcopy(base_dis_conf)
dis_conf[MUTAG][MAX_NUM_EPOCHS] = 230

dis_conf[PROTEINS] = copy.deepcopy(base_dis_conf)
dis_conf[PROTEINS][MAX_NUM_EPOCHS] = 420
dis_conf[PROTEINS][DISCRIMINATOR_PARAMS][NUM_LAYERS] = 6

dis_conf[IMDB_BINARY] = copy.deepcopy(base_dis_conf)
dis_conf[IMDB_BINARY][MAX_NUM_EPOCHS] = 320
dis_conf[IMDB_BINARY][DISCRIMINATOR_PARAMS][NUM_LAYERS] = 6

dis_conf[NCI109] = copy.deepcopy(base_dis_conf)
