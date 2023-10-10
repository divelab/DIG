import json
import pickle
import argparse
import torch
from torch.utils.data import DataLoader
from dig.ggraph3D.dataset import QM93DGEN, collate_fn
from dig.ggraph3D.method import G_SphereNet
from dig.ggraph3D.evaluation import RandGenEvaluator


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./G_SphereNet/rand_gen.pth', help='The path to the saved model file')
parser.add_argument('--num_mols', type=int, default=1000, help='The number of molecule geometries to be generated')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

with open('config_dict.json') as f:
    conf = json.load(f)

runner = G_SphereNet()

if args.train:
    dataset = QM93DGEN()
    idxs = dataset.get_idx_split('rand_gen')
    train_set = dataset[idxs['train']]
    loader = DataLoader(train_set, batch_size=conf['batch_size'], shuffle=True, collate_fn=collate_fn)
    runner.train(loader, lr=conf['lr'], wd=conf['weight_decay'], max_epochs=conf['max_epochs'], model_conf_dict=conf['model'], checkpoint_path=None, save_interval=conf['save_interval'], save_dir='rand_gen')
else:
    with torch.no_grad():
        mol_dicts = runner.generate(model_conf_dict=conf['model'], checkpoint_path=args.model_path, n_mols=args.num_mols, chunk_size=conf['chunk_size'], num_min_node=conf['num_min_node'], num_max_node=conf['num_max_node'], temperature=conf['temperature'], focus_th=conf['focus_th'])
    evaluator = RandGenEvaluator()

    print('Evaluating chemical validity...')
    results = evaluator.eval_validity(mol_dicts)
    
    print('Evaluating MMD distances of bond length distributions...')
    with open('target_bond_lengths.dict','rb') as f:
        target_bond_dists = pickle.load(f)
    input_dict = {'mol_dicts': mol_dicts, 'target_bond_dists': target_bond_dists}
    results = evaluator.eval_bond_mmd(input_dict)
