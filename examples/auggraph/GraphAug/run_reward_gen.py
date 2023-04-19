# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import argparse
from examples.auggraph.GraphAug.conf.reward_gen_conf import reward_gen_conf
from dig.auggraph.method.GraphAug.runner_reward_gen import RunnerRewardGen
from dig.auggraph.method.GraphAug.constants import *

dataset_name = DatasetName.IMDB_BINARY
conf = reward_gen_conf[dataset_name]
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str, default='../../../dig/auggraph/dataset/tudatasets',
                    help='The directory with all graph datasets')
parser.add_argument('--reward_gen_results_path', type=str, default='./results/reward_gen_results',
                    help='The directory where reward gen states will be stored after each epoch.')
args = parser.parse_args()

runner_reward_gen = RunnerRewardGen(args.data_root_path, dataset_name, conf)
runner_reward_gen.train_test(args.reward_gen_results_path)
