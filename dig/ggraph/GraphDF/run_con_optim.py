import torch
import os
from con_optim import ConOptim
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='graphaf', choices=['graphaf', 'jt'], help='dataset name')
parser.add_argument('--model_dir', type=str, default='./saved_ckpts/con_optim/con_optim_graphaf.pth', help='The path to the saved model file')
parser.add_argument('--out_dir', type=str, default='con_optim_out', help='The path to save output results')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

if args.data == 'graphaf':
    data_file = '../datasets/zinc_800_graphaf.csv'
    from config.con_optim_graphaf_config import conf
elif args.data == 'jt':
    data_file = '../datasets/zinc_800_jt.csv'
    from config.con_optim_jt_config import conf
else:
    print('Only graphaf and jt datasets are supported!')
    exit()

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
runner = ConOptim(conf, data_file, args.out_dir)

if args.train:
    runner.reinforce()
else:
    metrics = runner.constrained_optimize(save_mol=True, num_max_node=conf['num_max_node_for_gen'], temperature=conf['temperature_for_gen'])