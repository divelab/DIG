# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import argparse
from dig.auggraph.method.GraphAug.runner_aug_cls import RunnerAugCls
from dig.auggraph.method.GraphAug.constants import *
from examples.auggraph.GraphAug.conf.generator_conf import generator_conf
from examples.auggraph.GraphAug.conf.aug_cls_conf import aug_cls_conf

dataset_name = DatasetName.IMDB_BINARY
conf = aug_cls_conf[dataset_name]
conf[GENERATOR_PARAMS] = generator_conf[dataset_name][GENERATOR_PARAMS]
generator_checkpoint = generator_conf[dataset_name][MAX_NUM_EPOCHS] - 1
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str, default='../../../dig/auggraph/dataset/tudatasets',
                    help='The directory with all graph datasets')
parser.add_argument('--aug_model_path', type=str,
                    default='./results/generator_results/{}/max_augs_{}/{}.pt'.format(dataset_name.value, conf[GENERATOR_PARAMS][MAX_NUM_AUG], str(generator_checkpoint).zfill(4)),
                    help='The directory where generator states are stored after each epoch.')
parser.add_argument('--aug_cls_results_path', type=str,
                    default='./results/aug_cls_results/{}'.format(conf[MODEL_NAME].value),
                    help='The directory where classification results will be stored after each epoch.')
args = parser.parse_args()
conf[AUG_MODEL_PATH] = args.aug_model_path

runner = RunnerAugCls(args.data_root_path, dataset_name, conf)
for _ in range(5):
    runner.train_test(args.aug_cls_results_path)
