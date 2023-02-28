from examples.auggraph.GraphAug.conf.reward_gen_conf import reward_gen_conf
from dig.auggraph.method.GraphAug.runner_reward_gen import RunnerRewardGen
from dig.auggraph.method.GraphAug.constants import *
from examples.auggraph.GraphAug.conf.paths import *

dataset_name = DatasetName.IMDB_BINARY
conf = reward_gen_conf[dataset_name]

runner_reward_gen = RunnerRewardGen(DATA_ROOT_PATH, dataset_name, conf)
runner_reward_gen.train_test(REWARD_GEN_RESULTS_PATH)
