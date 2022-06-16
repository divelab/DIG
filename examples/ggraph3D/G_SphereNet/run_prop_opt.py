import json
import argparse
from torch.utils.data import DataLoader
from dig.ggraph3D.method import G_SphereNet
from dig.ggraph3D.evaluation import PropOptEvaluator
from dig.ggraph3D.dataset import QM93DGEN, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--prop', type=str, default='gap', choices=['gap', 'alpha'], help='property name')
parser.add_argument('--model_path', type=str, default='./G_SphereNet/prop_opt_gap.pth', help='The path to the saved model file')
parser.add_argument('--num_mols', type=int, default=100, help='The number of molecule geometries to be generated')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

with open('config_dict.json') as f:
    conf = json.load(f)

runner = G_SphereNet()

if args.train:
    dataset = QM93DGEN()
    idxs = dataset.get_idx_split('{}_opt'.format(args.prop))
    train_set = dataset[idxs['train']]
    loader = DataLoader(train_set, batch_size=conf['batch_size'], shuffle=True, collate_fn=collate_fn)
    runner.train(loader, lr=conf['lr'], wd=conf['weight_decay'], max_epochs=conf['max_epochs'], model_conf_dict=conf['model'], checkpoint_path=None, save_interval=conf['save_interval'], save_dir='prop_opt_{}'.format(args.prop))
else:
    mol_dicts = runner.generate(model_conf_dict=conf['model'], checkpoint_path=args.model_path, n_mols=args.num_mols, chunk_size=conf['chunk_size'], num_min_node=conf['num_min_node'], num_max_node=conf['num_max_node'], temperature=conf['temperature'], focus_th=conf['focus_th'])
    good_threshold = 4.5 if args.prop == 'gap' else 91
    evaluator = PropOptEvaluator(prop_name=args.prop, good_threshold=good_threshold)

    print('Evaluating...')
    results = evaluator.eval(mol_dicts)