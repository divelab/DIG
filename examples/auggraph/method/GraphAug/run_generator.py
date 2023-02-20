from examples.auggraph.method.GraphAug.conf.generator_conf import generator_conf
from examples.auggraph.method.GraphAug.conf.reward_gen_conf import reward_gen_conf
from dig.auggraph.method.GraphAug.runner_generator import RunnerGenerator
from dig.auggraph.method.GraphAug.constants import *
from dig.auggraph.method.GraphAug.paths import get_reward_gen_state_path

dataset_name = DatasetName.IMDB_BINARY
conf = generator_conf[dataset_name]
conf[REWARD_GEN_PARAMS] = reward_gen_conf[dataset_name][REWARD_GEN_PARAMS]
conf[REWARD_GEN_STATE_PATH] = get_reward_gen_state_path(dataset_name, conf[REWARD_GEN_PARAMS][MODEL_TYPE],
                                                        reward_gen_conf[dataset_name][MAX_NUM_EPOCHS] - 1)

runner = RunnerGenerator(dataset_name, conf)
runner.train()
