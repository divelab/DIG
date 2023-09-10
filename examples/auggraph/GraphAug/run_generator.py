# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import argparse
from examples.auggraph.GraphAug.conf.generator_conf import generator_conf
from examples.auggraph.GraphAug.conf.reward_gen_conf import reward_gen_conf
from dig.auggraph.method.GraphAug.runner_generator import RunnerGenerator
from dig.auggraph.method.GraphAug.constants import *

dataset_name = DatasetName.IMDB_BINARY
conf = generator_conf[dataset_name]
conf[REWARD_GEN_PARAMS] = reward_gen_conf[dataset_name][REWARD_GEN_PARAMS]
model_type = conf[REWARD_GEN_PARAMS][MODEL_TYPE]
last_checkpoint = reward_gen_conf[dataset_name][MAX_NUM_EPOCHS] - 1
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str, default='../../../dig/auggraph/dataset/tudatasets',
                    help='The directory with all graph datasets')
parser.add_argument('--generator_results_path', type=str,
                    default='./results/generator_results',
                    help='The directory where generator states will be stored after each epoch.')
parser.add_argument('--reward_gen_state_path', type=str,
                    default='./results/reward_gen_results/{}/{}/{}.pt'.format(dataset_name.value, model_type.value, str(last_checkpoint).zfill(4)),
                    help='File path for final training state of reward generation model')
args = parser.parse_args()
conf[REWARD_GEN_STATE_PATH] = args.reward_gen_state_path

runner = RunnerGenerator(args.data_root_path, dataset_name, conf)
runner.train_test(args.generator_results_path)
