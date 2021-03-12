import argparse
import time
import sys
import os
from tqdm import tqdm
import random

import numpy as np

import torch
from texttable import Texttable
from torch.utils.data import DataLoader, random_split, Subset
from distutils.util import strtobool
from rdkit.Chem import Draw
import cairosvg
from rdkit.Chem.Descriptors import qed

from preprocess_data import transform_qm9, transform_zinc250k
from model import *
from preprocess_data.data_loader import NumpyTupleDataset
from util import *

import sys
sys.path.append('..')
from utils import metric_random_generation, check_chemical_validity, qed, calculate_min_plogp


### Args
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./preprocess_data', help='Location for the dataset')
parser.add_argument('--property_name', type=str, default='qed', choices=['qed', 'plogp'], help='Property name')
parser.add_argument('--depth', type=int, default=2, help='Number of graph conv layers')
parser.add_argument('--add_self', type=strtobool, default='false', help='Add shortcut in graphconv')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimension')
parser.add_argument('--swish', type=strtobool, default='true', help='Use swish as activation function')
parser.add_argument('--c', type=float, default=0.5, help='Dequantization using uniform distribution of [0,c)')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size during training')
parser.add_argument('--model_qed_dir', type=str, help='Location for loading checkpoints')
parser.add_argument('--model_plogp_dir', type=str, help='Location for loading checkpoints')
parser.add_argument('--runs', type=int, default=5, help='# of runs')
parser.add_argument('--step_size', type=int, default=10, help='Step size in Langevin dynamics')
parser.add_argument('--sample_step', type=int, default=30, help='Number of sample step in Langevin dynamics')
parser.add_argument('--noise', type=float, default=0.005, help='The standard variance of the added noise during Langevin Dynamics')
parser.add_argument('--clamp', type=strtobool, default='true', help='Clamp the data/gradient during Langevin Dynamics')
parser.add_argument('--save_result_file', type=str, default=None, help='Save evaluation result')
parser.add_argument('--save_smiles', type=strtobool, default='true', help='Save the SMILES strings of generated melucules')
parser.add_argument('--save_fig', type=str, default=None, help='Save the drawn figs of generated melucules? If yes, give a directory.')

args = parser.parse_args()

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

tab_printer(args)


### Code adapted from https://github.com/rosinality/igebm-pytorch

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

        
def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def generate(model_qed, model_plogp, n_atom, n_atom_type, n_edge_type, device):
    parameters_qed = model_qed.parameters()
    parameters_plogp = model_plogp.parameters()
    ### Langevin dynamics
    gen_x = torch.rand(args.batch_size, n_atom, n_atom_type, device=device) * (1 + args.c) # (10000, 9, 5)
    gen_adj = torch.rand(args.batch_size, n_edge_type, n_atom, n_atom, device=device) #(10000, 4, 9, 9)
    
        
    gen_x.requires_grad = True
    gen_adj.requires_grad = True
    


    requires_grad(parameters_qed, False)
    requires_grad(parameters_plogp, False)
    model_qed.eval()
    model_plogp.eval()
    
    noise_x = torch.randn(gen_x.shape[0], n_atom, n_atom_type, device=device)  # (10000, 9, 5)
    noise_adj = torch.randn(gen_adj.shape[0], n_edge_type, n_atom, n_atom, device=device)  #(10000, 4, 9, 9) 

    for k in range(args.sample_step):

        noise_x.normal_(0, args.noise)
        noise_adj.normal_(0, args.noise)
        gen_x.data.add_(noise_x.data)
        gen_adj.data.add_(noise_adj.data)
        

        gen_out_qed = model_qed(gen_adj, gen_x)
        gen_out_plogp = model_plogp(gen_adj, gen_x)
        gen_out = 0.5 * gen_out_qed + 0.5 * gen_out_plogp
        gen_out.sum().backward()
        if args.clamp:
            gen_x.grad.data.clamp_(-0.01, 0.01)
            gen_adj.grad.data.clamp_(-0.01, 0.01)


        gen_x.data.add_(gen_x.grad.data, alpha=-args.step_size)
        gen_adj.data.add_(gen_adj.grad.data, alpha=-args.step_size)

        gen_x.grad.detach_()
        gen_x.grad.zero_()
        gen_adj.grad.detach_()
        gen_adj.grad.zero_()

        gen_x.data.clamp_(0, 1 + args.c)
        gen_adj.data.clamp_(0, 1)

    gen_x = gen_x.detach()
    gen_adj = gen_adj.detach()
    gen_adj = gen_adj + gen_adj.permute(0, 1, 3, 2)    # A+A^T is a symmetric matrix
    gen_adj = gen_adj / 2
    
   
    return gen_x, gen_adj   # (10000, 9, 5), (10000, 4, 9, 9)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_start = time.time()

    ### Load dataset
    if args.data_name=="qm9":
        atomic_num_list = [6, 7, 8, 9, 0]
        n_atom_type = 5
        n_atom = 9
        n_edge_type = 4
    elif args.data_name=="zinc250k":
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list
        n_atom_type = len(atomic_num_list) #10
        n_atom = 38
        n_edge_type = 4
    else:
        print("This dataset name is not supported!")


    ### Load trained model
    model_qed = GraphEBM(n_atom_type, args.hidden, n_edge_type, args.swish, args.depth, add_self = args.add_self)
    model_plogp = GraphEBM(n_atom_type, args.hidden, n_edge_type, args.swish, args.depth, add_self = args.add_self)
    print("Loading hyperparamaters from {}".format(args.model_qed_dir))
    model_qed.load_state_dict(torch.load(args.model_qed_dir))
    model_qed = model_qed.to(device)
    print("Loading paramaters from {}".format(args.model_plogp_dir))
    model_plogp.load_state_dict(torch.load(args.model_plogp_dir))
    model_plogp = model_plogp.to(device)
    

    t_end = time.time()

    print('Load trained model and data done! Time {:.2f} seconds'.format(t_end - t_start))
    print('==========================================')



    
    ### Random generation
    gen_time = []
    valid_ratio = []
    
    t_start = time.time()
    gen_x, gen_adj = generate(model_qed, model_plogp, n_atom, n_atom_type, n_edge_type, device)

    gen_mols = gen_mol(gen_adj, gen_x, atomic_num_list, correct_validity=True)
    gen_results = metric_random_generation(gen_mols)

    t_end = time.time()

    gen_time.append(t_end - t_start)

    valid_ratio.append(gen_results['valid_ratio'])
    
    valid_mols = [mol for mol in gen_mols if check_chemical_validity(mol)]

    prop_scores_qed = [qed(mol) for mol in valid_mols]
    prop_scores_plogp = [calculate_min_plogp(mol) for mol in valid_mols]

        
    if args.save_result_file is not None:
        file = args.save_result_file
        with open(file, 'a+') as f:
            f.write('Model: '+args.model_qed_dir+'\n')
            f.write('Model: '+args.model_plogp_dir+'\n')
            f.write('Validity: '+str(np.mean(valid_ratio))+'±'+str(np.std(valid_ratio))+'\n')
            f.write('Average qed score: '+ str(np.mean(prop_scores_qed)) + '±' + str(np.std(prop_scores_qed)) + '\n')
            f.write('Average plogp score: '+ str(np.mean(prop_scores_plogp)) + '±' + str(np.std(prop_scores_plogp)) + '\n')
            if args.save_smiles:
                f.write(str([Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid_mols])+'\n')
                f.write('\n')
                f.write(str(prop_scores_qed)+'\n')
                f.write('\n')
                f.write(str(prop_scores_plogp)+'\n')
            f.write('\n')
        
    if args.save_fig is not None:
        gen_dir = args.save_fig
        os.makedirs(gen_dir, exist_ok=True)
        for i in range(len(gen_mols)):
            filepath = os.path.join(gen_dir, 'generated_mols_{}.png'.format(i+1))
            img = Draw.MolToImage(gen_mols[i])
            img.save(filepath)


    print('==========================================')
    print('Average qed score:', str(np.mean(prop_scores_qed)), '±', str(np.std(prop_scores_qed)))
    print('Average plogp score:', str(np.mean(prop_scores_plogp)), '±', str(np.std(prop_scores_plogp)))
    print('==========================================')
    print("Validity: {:.3f}% ± {:.3f}%, vals={}".format(np.mean(valid_ratio), np.std(valid_ratio), valid_ratio))
    print('------------------------------------------')
    print("Generation Time: {:.3f} ± {:.3f} seconds, vals={}".format(np.mean(gen_time), np.std(gen_time), gen_time))
    print('==========================================')
    

        

    


            

    
