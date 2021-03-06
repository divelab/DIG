import torch
import numpy as np
from spherenet import spherenet
from torch_geometric.nn.acts import swish
import argparse
import torch.nn.functional as F
from train import run
import sys
sys.path.append('..')
from utils import load_qm9, load_md17

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="benzene", help='dataset name') # qm9, aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic, toluene, uracil
parser.add_argument('--target', type=str, default='U0')
parser.add_argument('--num_task', type=int, default=1)
parser.add_argument('--output_init', type=str, default='GlorotOrthogonal') # in DimeNet++, 'zeros' for mu, homo, lumo, and zpve; 'GlorotOrthogonal' for alpha, R2, U0, U, H, G, and Cv

parser.add_argument('--save_dir', type=str, default='../trained_models/', help='directory to save the model with best validation performance')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.96)
parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--energy_and_force', type=bool, default=False)
parser.add_argument('--num_atom', type=int, default=1)
parser.add_argument('--p', type=int, default=100)

parser.add_argument('--cutoff', type=float, default=5.0) #5.0
parser.add_argument('--num_layer', type=int, default=4)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--int_emb_size', type=int, default=64)
parser.add_argument('--basis_emb_size', type=int, default=8)
parser.add_argument('--out_emb_size', type=int, default=256)
parser.add_argument('--num_spherical', type=int, default=7)
parser.add_argument('--num_radial', type=int, default=6)

args = parser.parse_args()

if args.dataset == 'qm9':
    train_size = 110000
    val_size = 10000
    train_dataset, val_dataset, test_dataset = load_qm9(args.dataset, args.target, train_size, val_size)

elif args.dataset in ['aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil']: #md17
    train_size = 1000
    val_size = 10000
    train_dataset, val_dataset, test_dataset, num_atom = load_md17(args.dataset, train_size, val_size)
    args.energy_and_force = True
    args.num_atom = num_atom

else:
    print("This dataset name is not supported!")

print(args)
model = spherenet(args.cutoff, args.num_layer, args.hidden_channels, args.num_task, args.int_emb_size, args.basis_emb_size, args.out_emb_size, args.num_spherical, args.num_radial, output_init=args.output_init)
run(train_dataset, val_dataset, model, args.epochs, args.batch_size, args.lr, args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay, args.save_dir, args.energy_and_force, args.num_atom, args.p)