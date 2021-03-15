import torch
import os
from ran_gen import RandomGen
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import sys
sys.path.append('..')
from utils import get_smiles_qm9, get_smiles_zinc250k, get_smiles_moses
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='qm9', choices=['qm9', 'zinc250k', 'moses'], help='dataset name')
parser.add_argument('--model_dir', type=str, default='./saved_ckpts/ran_gen/ran_gen_qm9.pth', help='The path to the saved model file')
parser.add_argument('--out_dir', type=str, default='ran_gen_out', help='The path to save output results')
parser.add_argument('--num_mols', type=int, default=100, help='The number of molecules to be generated')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

if args.data == 'qm9':
    smiles = get_smiles_qm9('../datasets/qm9.csv')
    from config.ran_gen_qm9_config import conf
elif args.data == 'zinc250k':
    smiles = get_smiles_zinc250k('../datasets/zinc250k.csv')
    from config.ran_gen_zinc250k_config import conf
elif args.data == 'moses':
    smiles = get_smiles_moses('../datasets/moses.csv')
    from config.ran_gen_moses_config import conf
else:
    print("Only qm9, zinc250k and moses datasets are supported!")
    exit()

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
runner = RandomGen(conf, smiles, args.out_dir)

if args.train:
    runner.train()
else:
    runner.model.load_state_dict(torch.load(args.model_dir))
    unique_rate, valid_no_check_rate, valid_rate, novelty_rate, all_mols = runner.evaluate(num=args.num_mols, save_mols=True, verbose=True,
        num_min_node=conf['num_min_node_for_gen'], num_max_node=conf['num_max_node_for_gen'], temperature=conf['temperature_for_gen'])