# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import copy
from dig.auggraph.method.GraphAug.constants import *

base_generator_conf = {
    BATCH_SIZE: 32,
    INITIAL_LR: 1e-4,
    GENERATOR_STEPS: 1,
    TEST_INTERVAL: 1,
    MAX_NUM_EPOCHS: 200,
    REWARD_GEN_STATE_PATH: None,
    BASELINE: BaselineType.MEAN,
    MOVING_RATIO: 0.1,
    SAVE_MODEL: True,
    GENERATOR_PARAMS: {
        NUM_LAYERS: 3,
        HID_DIM: 64,
        MAX_NUM_AUG: 8,
        USE_STOP_AUG: False,
        UNIFORM: False,
        RNN_INPUT: RnnInputType.VIRTUAL,
        AUG_TYPE_PARAMS: {
            AugType.NODE_FM.value: {
                HID_DIM: 64, TEMPERATURE: 1.0, TRAINING: True, MAGNITUDE: 0.05
            },
            AugType.NODE_DROP.value: {
                HID_DIM: 64, TEMPERATURE: 1.0, TRAINING: True, MAGNITUDE: 0.05
            },
            AugType.EDGE_Per.value: {
                HID_DIM: 64, TEMPERATURE: 1.0, TRAINING: True, MAGNITUDE: 0.05
            }
        }
    }
}

generator_conf = {}

generator_conf[DatasetName.NCI1] = copy.deepcopy(base_generator_conf)

generator_conf[DatasetName.COLLAB] = copy.deepcopy(base_generator_conf)
generator_conf[DatasetName.COLLAB][BATCH_SIZE] = 8

generator_conf[DatasetName.NCI109] = copy.deepcopy(base_generator_conf)

generator_conf[DatasetName.MUTAG] = copy.deepcopy(base_generator_conf)
generator_conf[DatasetName.MUTAG][BATCH_SIZE] = 16

generator_conf[DatasetName.PROTEINS] = copy.deepcopy(base_generator_conf)
generator_conf[DatasetName.PROTEINS][GENERATOR_PARAMS][NUM_LAYERS] = 6

generator_conf[DatasetName.IMDB_BINARY] = copy.deepcopy(base_generator_conf)
generator_conf[DatasetName.IMDB_BINARY][GENERATOR_PARAMS][NUM_LAYERS] = 6
