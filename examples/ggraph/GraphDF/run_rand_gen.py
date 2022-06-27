import json
import argparse
from rdkit import RDLogger
from torch_geometric.loader import DenseDataLoader
from dig.ggraph.dataset import QM9, ZINC250k, MOSES
from dig.ggraph.method import GraphDF
from dig.ggraph.evaluation import RandGenEvaluator

RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='qm9', choices=['qm9', 'zinc250k', 'moses'], help='dataset name')
parser.add_argument('--model_path', type=str, default='./saved_ckpts/rand_gen/rand_gen_qm9.pth', help='The path to the saved model file')
parser.add_argument('--num_mols', type=int, default=100, help='The number of molecules to be generated')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

if args.data == 'qm9':
    with open('config/rand_gen_qm9_config_dict.json') as f:
        conf = json.load(f)
    dataset = QM9(conf_dict=conf['data'], one_shot=False, use_aug=True)
elif args.data == 'zinc250k':
    with open('config/rand_gen_zinc250k_config_dict.json') as f:
        conf = json.load(f)
    dataset = ZINC250k(conf_dict=conf['data'], one_shot=False, use_aug=True)
elif args.data == 'moses':
    with open('config/rand_gen_moses_config_dict.json') as f:
        conf = json.load(f)
    dataset = MOSES(conf_dict=conf['data'], one_shot=False, use_aug=True)  
else:
    print("Only qm9, zinc250k and moses datasets are supported!")
    exit()

runner = GraphDF()

if args.train:
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    runner.train_rand_gen(loader, conf['lr'], conf['weight_decay'], conf['max_epochs'], conf['model'], conf['save_interval'], conf['save_dir'])
else:
    mols, pure_valids = runner.run_rand_gen(conf['model'], args.model_path, args.num_mols, conf['num_min_node'], conf['num_max_node'], conf['temperature'], conf['atom_list'])
    smiles = [data.smile for data in dataset]
    evaluator = RandGenEvaluator()
    input_dict = {'mols': mols, 'train_smiles': smiles}

    print('Evaluating...')
    results = evaluator.eval(input_dict)

    print("Valid Ratio without valency check: {:.2f}%".format(sum(pure_valids) / args.num_mols * 100))