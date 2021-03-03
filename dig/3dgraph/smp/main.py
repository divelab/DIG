import os.path as osp
import argparse

import torch
import numpy as np
import pickle

from torch_geometric.datasets import QM9, QM8, QM7b, QM7
from torch_geometric.data import DataLoader
from schnet import SchNet
from dimenet import DimeNet
from dimenetpp import DimeNetPlusPlus
from dimenetppwithtorsion import DimeNetPlusPluswithtorsion
from torch_geometric.nn.acts import swish

import argparse
import torch
import torch.nn.functional as F
# from texttable import Texttable
import sys

from datasets import *
from train_eval import run_classification, run_regression
from torch_scatter import scatter_mean
from schnet import SchNet
from dimenet import DimeNet
from dimenetpp import DimeNetPlusPlus

from sklearn.utils import shuffle
from torch_geometric.nn.acts import swish


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="qm9", help='dataset name')

### If split ready is True, use (1); Otherwise, use (2).###
parser.add_argument('--split_ready', action='store_true', default=False, help='specify it to be true if you provide three files for train/val/test')
####################################################################

### (1) The following arguments are used when split_ready==True.###
parser.add_argument('--trainfile', type=str, help='path to the preprocessed training file (Pytorch Geometric Data)')
parser.add_argument('--validfile', type=str, help='path to the preprocessed validation file (Pytorch Geometric Data)')
parser.add_argument('--testfile', type=str, help='path to the preprocessed test file (Pytorch Geometric Data)')
####################################################################

### (2) The following arguments are used when split_ready==False.###
parser.add_argument('--ori_dataset_path', type=str, default="../../datasets/moleculenet/", help='directory of the original csv file (SMILES string)')
parser.add_argument('--pro_dataset_path', type=str, default="DIG/dig/3dgraph/data/", help='directory of the preprocessed data (Pytorch Geometric Data)')
parser.add_argument('--split_file_path', type=str, default="data/split_inds/")
parser.add_argument('--split_mode', type=str, default='random', help=' split methods, use random, stratified or scaffold')
parser.add_argument('--split_train_ratio', type=float, default=0.8, help='the ratio of data for training set')
parser.add_argument('--split_valid_ratio', type=float, default=0.1, help='the ratio of data for validation set')
parser.add_argument('--seed', type=int, default=122, help='random seed for split, use 122, 123 or 124')
####################################################################

parser.add_argument('--log_dir', type=str, default=None, help='directory to save train/val information') 
parser.add_argument('--save_dir', type=str, default='../trained_models/your_model/', help='directory to save the model with best validation performance')
parser.add_argument('--evaluate', action='store_true', default=False, help='specify it to be true if you want to do evaluation on test set (available only when test labels are provided)')

parser.add_argument('--model', type=str, default="dimenetpp")
parser.add_argument('--skip_qm9', type=bool, default=True, help='specify it to be true if skip 3054 data in QM9 (like data in DimeNet)')
parser.add_argument('--target', type=list, default=['Cv'])
parser.add_argument('--output_init', type=str, default='zeros') # in DimeNet++, 'zeros' for mu, homo, lumo, and zpve; 'GlorotOrthogonal' for alpha, R2, U0, U, H, G, and Cv

parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.96) #0.5
parser.add_argument('--lr_decay_step_size', type=int, default=1) #50

# parser.add_argument('--gpu_ids', type=str, default='', help='which gpus to use, one or multiple')

args = parser.parse_args()

# if len(args.gpu_ids) > 0:
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

# def tab_printer(args):
#     args = vars(args)
#     keys = sorted(args.keys())
#     t = Texttable() 
#     t.set_precision(10)
#     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
#     print(t.draw())

# tab_printer(args)


#sys.path.append("..")
confs = __import__('config_default', fromlist=['conf'])
conf = confs.conf
conf['model'] = args.model
conf['output_init'] = args.output_init
conf['num_tasks'] = len(args.target)
conf['epochs'] = args.epochs
conf['early_stopping'] = args.early_stopping
conf['lr'] = args.lr
conf['lr_decay_factor'] = args.lr_decay_factor
conf['lr_decay_step_size'] = args.lr_decay_step_size
print(conf)

qm9_dimenet = {'mu':3, 'alpha':4, 'homo':5, 'lumo':6, 'gap':7, 'r2':8, 'zpve':9, 'U0':15, 'U':16, 'H':17, 'G':18, 'Cv':14}
qm9_dimenet_weight = torch.tensor([1, 1, 1, 1, 1, 27.2113825435, 27.2113825435, 27.2113825435, 1, 27.2113825435, 1, 1, 1, 1, 1, 0.04336414, 0.04336414, 0.04336414, 0.04336414])

qm9_moleculenet = {'mu':3, 'alpha':4, 'homo':5, 'lumo':6, 'gap':7, 'r2':8, 'zpve':9, 'cv':14, 'u0':10, 'u298':11, 'h298':12, 'g298':13}

print(args.target)

if args.dataset == 'qm7':
    dataset = torch.load(args.pro_dataset_path+args.dataset+'.pt') # /mnt/dive/shared/limei/graph/AICURES/DIG/dig/3dgraph/data/qm7.pt
    f = open(args.split_file_path+'qm7stratified'+str(args.seed)+'.pkl', 'rb') # '/mnt/dive/shared/limei/graph/AICURES/DIG/dig/3dgraph/data/split_inds/qm7stratified122.pkl
    train_idx, val_idx, test_idx = pickle.load(f, encoding='bytes')
    f.close()
elif args.dataset == 'qm8':
    dataset = torch.load(args.pro_dataset_path+args.dataset+'.pt') # /mnt/dive/shared/limei/graph/AICURES/DIG/dig/3dgraph/data/qm8.pt
    f = open(args.split_file_path+'qm8random'+str(args.seed)+'.pkl', 'rb')
    train_idx, val_idx, test_idx = pickle.load(f, encoding='bytes')
    f.close()
elif args.dataset == 'qm9':
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    # dataset = QM9(path)
    if args.skip_qm9:
        dataset = torch.load(args.pro_dataset_path+args.dataset+'_skip.pt') # (data in DimeNet) /mnt/dive/shared/limei/graph/AICURES/DIG/dig/3dgraph/data/qm9_skip.pt
        indices = torch.tensor([qm9_dimenet[target] for target in args.target])
        for data in dataset:
            data.y[0] = data.y[0] * qm9_dimenet_weight
            data.y = torch.index_select(data.y, 1, indices)
        ids = shuffle(range(130831), random_state=42)
        train_idx, val_idx, test_idx = np.array(ids[:110000]), np.array(ids[110000:120000]), np.array(ids[120000:])
    else:
        dataset = torch.load(args.pro_dataset_path+args.dataset+'.pt') # (data in MoleculeNet) /mnt/dive/shared/limei/graph/AICURES/DIG/dig/3dgraph/data/qm9.pt
        indices = torch.tensor([qm9_moleculenet[target] for target in args.target])
        for data in dataset:
            data.y = torch.index_select(data.y, 1, indices)   
        f = open(args.split_file_path+'qm9random'+str(args.seed)+'.pkl', 'rb')
        train_idx, val_idx, test_idx = pickle.load(f, encoding='bytes')
        f.close()

train_dataset = [dataset[int(i)] for i in train_idx]
val_dataset = [dataset[int(i)] for i in val_idx]
test_dataset = [dataset[int(i)] for i in test_idx]


### Choose model
if conf['model'] == "schnet":
    model = SchNet(hidden_channels = conf['schnet_hidden_channels'], num_filters = conf['num_filters'], 
    num_interactions = conf['num_interactions'], num_gaussians = conf['num_gaussians'], cutoff = conf['schnet_cutoff'], readout='add')
elif conf['model'] == "dimenet":
    model = DimeNet(conf['dimenet_hidden_channels'], conf['num_tasks'], conf['num_blocks'], conf['num_bilinear'],
                 conf['num_spherical'], conf['num_radial'], conf['dimenet_cutoff'], conf['envelope_exponent'],
                 conf['num_before_skip'], conf['num_after_skip'], conf['num_output_layers'])
elif conf['model'] == "dimenetpp":
    model = DimeNetPlusPlus(conf['dimenet_hidden_channels'], conf['num_tasks'], conf['num_blocks'], conf['int_emb_size'], conf['basis_emb_size'], conf['out_emb_channels'],
                 conf['num_spherical'], conf['num_radial'], conf['dimenet_cutoff'], conf['envelope_exponent'],
                 conf['num_before_skip'], conf['num_after_skip'], conf['num_output_layers'], act=swish, output_init=conf['output_init'])
elif conf['model'] == "dimenetpptorsion":
    model = DimeNetPlusPluswithtorsion(conf['dimenet_hidden_channels'], conf['num_tasks'], conf['num_blocks'], conf['int_emb_size'], conf['basis_emb_size'], conf['out_emb_channels'],
                 conf['num_spherical'], conf['num_radial'], conf['dimenet_cutoff'], conf['envelope_exponent'],
                 conf['num_before_skip'], conf['num_after_skip'], conf['num_output_layers'], act=swish, output_init=conf['output_init'])
else:
    print('Please choose correct model!!!')

### Run
if conf['task_type'] == "classification":
    run_classification(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['early_stopping'], conf['metric'], args.log_dir, args.save_dir, args.evaluate)
elif conf['task_type'] == "regression":
    run_regression(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['early_stopping'], conf['metric'], args.log_dir, args.save_dir, args.evaluate)
else:
    print("Wrong task type!!!")