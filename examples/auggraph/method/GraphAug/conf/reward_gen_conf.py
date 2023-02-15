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
        MODEL_TYPE: GMNET,
        POOL_TYPE: SUM,
        FUSE_TYPE: ABS_DIFF
    }
}

reward_gen_conf = {}

reward_gen_conf[NCI1] = copy.deepcopy(base_reward_gen_conf)

reward_gen_conf[COLLAB] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[COLLAB][BATCH_SIZE] = 8
reward_gen_conf[COLLAB][MAX_NUM_EPOCHS] = 120

reward_gen_conf[MUTAG] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[MUTAG][MAX_NUM_EPOCHS] = 230

reward_gen_conf[PROTEINS] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[PROTEINS][MAX_NUM_EPOCHS] = 420
reward_gen_conf[PROTEINS][REWARD_GEN_PARAMS][NUM_LAYERS] = 6

reward_gen_conf[IMDB_BINARY] = copy.deepcopy(base_reward_gen_conf)
reward_gen_conf[IMDB_BINARY][MAX_NUM_EPOCHS] = 320
reward_gen_conf[IMDB_BINARY][REWARD_GEN_PARAMS][NUM_LAYERS] = 6

reward_gen_conf[NCI109] = copy.deepcopy(base_reward_gen_conf)
