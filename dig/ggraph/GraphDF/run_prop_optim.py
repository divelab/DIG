import torch
import os
from prop_optim import PropOptim
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prop', type=str, default='plogp', choices=['plogp', 'qed'], help='property name')
parser.add_argument('--model_dir', type=str, default='./saved_ckpts/prop_optim/prop_optim_plogp.pth', help='The path to the saved model file')
parser.add_argument('--out_dir', type=str, default='prop_optim_out', help='The path to save output results')
parser.add_argument('--num_mols', type=int, default=100, help='The number of molecules to be generated')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

if args.prop == 'plogp':
    from config.prop_optim_plogp_config import conf
elif args.prop == 'qed':
    from config.prop_optim_qed_config import conf
else:
    print('Only plogp and qed properties are supported!')
    exit()

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
runner = PropOptim(conf, args.out_dir)

if args.train:
    runner.reinforce()
else:
    runner.model.load_state_dict(torch.load(args.model_dir))
    top3 = runner.generate_molecule(num=args.num_mols, save_mols=True, verbose=True, 
        num_min_node=conf['num_min_node_for_gen'], num_max_node=conf['num_max_node_for_gen'], temperature=conf['temperature_for_gen'])