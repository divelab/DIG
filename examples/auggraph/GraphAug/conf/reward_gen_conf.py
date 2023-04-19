# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import copy
from dig.auggraph.method.GraphAug.constants import *

base_reward_gen_conf = {
    BATCH_SIZE: 32,
    INITIAL_LR: 1e-4,
    FACTOR: 0.5,
    PATIENCE: 5,
    MIN_LR: 1e-7,
    MAX_NUM_EPOCHS: 200,
    PRE_TRAIN_PATH: None,
    REWARD_GEN_PARAMS: {
        NUM_LAYERS: 5,
        HIDDEN_UNITS: 256,
        MODEL_TYPE: RewardGenModelType.GMNET,
        POOL_TYPE: PoolType.SUM,
        FUSE_TYPE: FuseType.ABS_DIFF
    }
}

reward_gen_conf = {}

reward_gen_conf[DatasetName.NCI1] = copy.deepcopy(base_reward_gen_conf)

reward_gen_conf[DatasetName.COLLAB] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[DatasetName.COLLAB][BATCH_SIZE] = 8
reward_gen_conf[DatasetName.COLLAB][MAX_NUM_EPOCHS] = 120

reward_gen_conf[DatasetName.MUTAG] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[DatasetName.MUTAG][MAX_NUM_EPOCHS] = 230

reward_gen_conf[DatasetName.PROTEINS] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[DatasetName.PROTEINS][MAX_NUM_EPOCHS] = 420
reward_gen_conf[DatasetName.PROTEINS][REWARD_GEN_PARAMS][NUM_LAYERS] = 6

reward_gen_conf[DatasetName.IMDB_BINARY] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[DatasetName.IMDB_BINARY][MAX_NUM_EPOCHS] = 320
reward_gen_conf[DatasetName.IMDB_BINARY][REWARD_GEN_PARAMS][NUM_LAYERS] = 6

reward_gen_conf[DatasetName.NCI109] = copy.deepcopy(base_reward_gen_conf)
