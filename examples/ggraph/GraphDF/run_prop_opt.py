import json
import argparse
from rdkit import RDLogger
from dig.ggraph.method import GraphDF
from dig.ggraph.evaluation import PropOptEvaluator

RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument('--prop', type=str, default='plogp', choices=['plogp', 'qed'], help='property name')
parser.add_argument('--model_path', type=str, default='./saved_ckpts/prop_opt/prop_opt_plogp.pth', help='The path to the saved model file')
parser.add_argument('--num_mols', type=int, default=100, help='The number of molecules to be generated')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

if args.prop == 'plogp':
    with open('config/prop_opt_plogp_config_dict.json') as f:
        conf = json.load(f)
elif args.prop == 'qed':
    with open('config/prop_opt_qed_config_dict.json') as f:
        conf = json.load(f)
else:
    print('Only plogp and qed properties are supported!')
    exit()

runner = GraphDF()

if args.train:
    runner.train_prop_opt(conf['lr'], conf['weight_decay'], conf['max_iters'], conf['warm_up'], conf['model'], conf['pretrain_model'], conf['save_interval'], conf['save_dir'])
else:
    mols = runner.run_prop_opt(conf['model'], args.model_path, args.num_mols, conf['num_min_node'], conf['num_max_node'], conf['temperature'], conf['atom_list'])
    evaluator = PropOptEvaluator()
    input_dict = {'mols': mols}

    print('Evaluating...')
    results = evaluator.eval(input_dict)