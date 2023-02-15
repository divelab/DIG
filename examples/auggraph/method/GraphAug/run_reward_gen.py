from examples.auggraph.method.GraphAug.conf.reward_gen_conf import reward_gen_conf
from dig.auggraph.method.GraphAug.runner_reward_gen import RunnerRewardGen
from dig.auggraph.method.GraphAug.constants import *

dataset_name = IMDB_BINARY

runner_reward_gen = RunnerRewardGen(dataset_name, reward_gen_conf[dataset_name])
runner_reward_gen.train_test()
