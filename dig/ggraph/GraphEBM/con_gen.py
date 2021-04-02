import argparse
import time
import sys
import os
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import copy


import torch
from texttable import Texttable
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from distutils.util import strtobool
from rdkit.Chem import Draw
import cairosvg
from rdkit.Chem.Descriptors import qed



from preprocess_data import transform_qm9, transform_zinc250k
from model import *
from preprocess_data.data_loader import NumpyTupleDataset
from util import *

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import sys
sys.path.append('..')
from utils import metric_random_generation, check_chemical_validity, qed, calculate_min_plogp

### Args
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='zinc250k', choices=['qm9', 'zinc250k'], help='dataset name')
parser.add_argument('--model_name', type=str, default='basic', choices=['basic', 'sketch'], help='Dataset name')
parser.add_argument('--data_dir', type=str, default='./preprocess_data', help='Location for the dataset')
parser.add_argument('--property_name', type=str, default='plogp', choices=['qed', 'plogp'], help='Property name')
parser.add_argument('--normal_adj', type=strtobool, default='true', help='Normalize the adjacency tensor')
parser.add_argument('--depth', type=int, default=2, help='Number of graph conv layers')
parser.add_argument('--add_self', type=strtobool, default='false', help='Add shortcut in graphconv')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimension')
parser.add_argument('--swish', type=strtobool, default='true', help='Use swish as activation function')
parser.add_argument('--c', type=float, default=0, help='Dequantization using uniform distribution of [0,c)')
parser.add_argument('--batch_size', type=int, default=800, help='Batch size during training')
parser.add_argument('--model_dir', type=str, default='./release_models/model_zinc250k_goal_plogp.pt', help='Location for loading checkpoints')
parser.add_argument('--runs', type=int, default=1, help='# of runs')
parser.add_argument('--step_size', type=float, default=0.2, help='Step size in Langevin dynamics')
parser.add_argument('--sample_step', type=int, default=500, help='Number of sample step in Langevin dynamics')
parser.add_argument('--correct_validity', type=strtobool, default='true', help='If apply validity correction after the generation')
parser.add_argument('--save_result_file', type=str, default='./result/cons_testset.txt', help='Save evaluation result')
parser.add_argument('--save_smiles', type=strtobool, default='true', help='If save generated melucules')
parser.add_argument('--save_fig', type=str, default=None, help='If save generated melucules')
parser.add_argument('--exp_name', type=str, default='goal_plogp_testset', help='If save generated melucules')

args = parser.parse_args()

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

tab_printer(args)

def list_of_tuples(tuple_of_list):
    assert(len(tuple_of_list)) == 3
    list1 = tuple_of_list[0]
    list2 = tuple_of_list[1]
    list3 = tuple_of_list[2]
    
    return list(map(lambda x, y, z: (x , y, z), list1, list2, list3))

def load_mol(filepath):
    print('Loading file {}'.format(filepath))
    if not os.path.exists(filepath):
        raise ValueError('Invalid filepath {} for dataset'.format(filepath))
    load_data = np.load(filepath)
    result = []  # a tuple of list ((133885), (133885), (133885))
    i = 0
    while True:
        key = 'arr_{}'.format(i)
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list_of_tuples(result)


class SimpleDataset(Dataset):
    def __init__(self, mols, y, transform):
        '''
        mols: a list of tuples. Each tuple: ((9,), (4,9,9), (15,))
        y: a list of scalar values
        '''
        self.mols = mols
        self.y = list(map(float, y))
        self.transform = transform
        assert len(self.mols) == len(self.y)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx]), np.expand_dims(self.y[idx], 0)


### Adapted from https://github.com/calvin-zcx/moflow/blob/master/mflow/optimize_property.py
def load_property_csv(data_name, property_name='qed', normalize=True):
    """
    We use qed and plogp in zinc250k_property.csv which are recalculated by rdkit
    the recalculated qed results are in tiny inconsistent with qed in zinc250k.csv
    e.g
    zinc250k_property.csv:
    qed,plogp,smile
    0.7319008436872337,3.1399057164163766,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
    0.9411116113894995,0.17238635659148804,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1
    import rdkit
    m = rdkit.Chem.MolFromSmiles('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1')
    rdkit.Chem.QED.qed(m): 0.7319008436872337
    from mflow.utils.environment import penalized_logp
    penalized_logp(m):  3.1399057164163766
    However, in oringinal:
    zinc250k.csv
    ,smiles,logP,qed,SAS
    0,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1,5.0506,0.702012232801,2.0840945720726807
    1,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1,3.1137,0.928975488089,3.4320038192747795
    0.7319008436872337 v.s. 0.702012232801
    and no plogp in zinc250k.csv dataset!
    """
    if data_name == 'qm9':
        filename = '../datasets/qm9_property.csv'
    elif data_name == 'zinc250k':
        filename = '../datasets/zinc250k_property.csv'

    df = pd.read_csv(filename)  # qed, plogp, smile
    if property_name=='plogp' and normalize:
        # plogp: # [-62.52, 4.52]
        m = df['plogp'].mean()  # 0.00026
        std = df['plogp'].std() # 2.05
        mn = df['plogp'].min()
        mx = df['plogp'].max()
        # df['plogp'] = 0.5 * (np.tanh(0.01 * ((df['plogp'] - m) / std)) + 1)  # [0.35, 0.51]
        # df['plogp'] = (df['plogp'] - m) / std
        lower = -10 # -5 # -10
        df['plogp'] = df['plogp'].clip(lower=lower, upper=5)
        df['plogp'] = (df['plogp'] - lower) / (mx-lower)

    if property_name == 'qed':
        col = 0
    elif property_name == 'plogp':
        col = 1
    else:
        raise ValueError("Wrong property_name{}".format(property_name))
    prop = [x[col] for x in df.values]
    print('Load {} done, length: {}'.format(filename, len(prop)))
    return prop    

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
                


def generate(model, init_dataloader, n_atom, n_atom_type, n_edge_type, device, atomic_num_list):
    optim_dict = dict()
    save_smiles_dict = dict()
    parameters = model.parameters()
    
    ### Langevin dynamics
    for i, batch in enumerate(tqdm(init_dataloader)):
        gen_x = batch[0][0].to(device) 
        gen_adj = batch[0][1].to(device)
        print(batch[0][0].shape)
        original_mols = turn_valid(gen_adj, gen_x, atomic_num_list, correct_validity=args.correct_validity)
        
        gen_x.requires_grad = True
        gen_adj.requires_grad = True

        requires_grad(parameters, False)
        model.eval()
    
        noise_x = torch.randn(gen_x.shape[0], n_atom, n_atom_type, device=device)  # (10000, 9, 5)
        noise_adj = torch.randn(gen_adj.shape[0], n_edge_type, n_atom, n_atom, device=device)  #(10000, 4, 9, 9) 

        for k in tqdm(range(args.sample_step)):

            noise_x.normal_(0, 0.005)
            noise_adj.normal_(0, 0.005)
            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)
            

            gen_out = model(gen_adj, gen_x)
            gen_out.sum().backward()
            gen_x.grad.data.clamp_(-0.1, 0.1)
            gen_adj.grad.data.clamp_(-0.1, 0.1)


            gen_x.data.add_(-args.step_size, gen_x.grad.data)
            gen_adj.data.add_(-args.step_size, gen_adj.grad.data)

            gen_x.grad.detach_()
            gen_x.grad.zero_()
            gen_adj.grad.detach_()
            gen_adj.grad.zero_()
            
            gen_x.data.clamp_(0, 1 + args.c)
            gen_adj.data.clamp_(0, 1)

            
            # if k % 2 == 0:
            gen_x_t = copy.deepcopy(gen_x)
            gen_adj_t = copy.deepcopy(gen_adj)
            gen_adj_t = gen_adj_t + gen_adj_t.permute(0, 1, 3, 2)    # A+A^T is a symmetric matrix
            gen_adj_t = gen_adj_t / 2
            gen_adj_t = gen_adj_t.softmax(dim=1)   ### Make all elements to be larger than 0
            max_bond = gen_adj_t.max(dim=1).values.reshape(args.batch_size, -1, n_atom, n_atom)  # (10000, 1, 9, 9)
            gen_adj_t = torch.floor(gen_adj_t / max_bond)  # (10000, 4, 9, 9) /  (10000, 1, 9, 9) -->  (10000, 4, 9, 9)
            val_res = turn_valid(gen_adj_t, gen_x_t, atomic_num_list, correct_validity=args.correct_validity)
            assert len(val_res['valid_mols']) == len(original_mols['valid_mols'])
            
            for mol_idx in range(len(val_res['valid_mols'])):
                if val_res['valid_mols'][mol_idx] is not None:
                    tmp_mol = val_res['valid_mols'][mol_idx]
                    tmp_smiles = val_res['valid_smiles'][mol_idx]
                    o_mol = original_mols['valid_mols'][mol_idx]
                    o_smiles = original_mols['valid_smiles'][mol_idx]
                    # calculate imp
                    if args.property_name=='qed':
                        imp_p = qed(tmp_mol) - qed(o_mol)
                    elif args.property_name=='plogp':
                        try:
                            imp_p = calculate_min_plogp(tmp_mol) - calculate_min_plogp(o_mol)
                            # calculate sim
                            current_sim = reward_target_molecule_similarity(tmp_mol, o_mol)
                            update_optim_dict(optim_dict, o_smiles, tmp_smiles, imp_p, current_sim)
                            update_save_dict(save_smiles_dict, o_smiles, tmp_smiles, imp_p, current_sim)
                        except:
                            # print('plogp calculate error!')
                            pass
                        


    return optim_dict, save_smiles_dict 




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_start = time.time()

    ### Load dataset
    if args.data_name=="qm9":
        data_file = "qm9_relgcn_kekulized_ggnp.npz"
        transform_fn = transform_qm9.transform_fn
        atomic_num_list = [6, 7, 8, 9, 0]
        file_path = '../datasets/valid_idx_qm9.json'
        valid_idx = transform_qm9.get_val_ids(file_path)
        n_atom_type = 5
        n_atom = 9
        n_edge_type = 4
    elif args.data_name=="zinc250k":
        data_file = "zinc250k_relgcn_kekulized_ggnp.npz"
        transform_fn = transform_zinc250k.transform_fn
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list
        file_path = '../datasets/valid_idx_zinc250k.json'
        valid_idx = transform_zinc250k.get_val_ids(file_path)
        n_atom_type = len(atomic_num_list) #10
        n_atom = 38
        n_edge_type = 4
    else:
        print("This dataset name is not supported!")
    
    mols = load_mol(os.path.join(args.data_dir, data_file)) # 133885
    train_idx = [t for t in range(len(mols)) if t not in valid_idx]
    train_mols = [mols[t] for t in train_idx]  # 120803 = 133885-13082
    test_mols = [mols[t] for t in valid_idx]  # 13,082
    
    prop_list = load_property_csv(args.data_name, args.property_name)
    train_prop = [prop_list[i] for i in train_idx]
    test_prop = [prop_list[i] for i in valid_idx]

    # smallest 800 plogp molecules in test set
    inds = sorted(range(len(test_prop)), key=lambda k: test_prop[k], reverse=False)
    test_smiles = [test_mols[i] for i in inds]
    test_scores = [test_prop[i] for i in inds]
    # print(test_scores[:800])

    # smallest 800 plogp molecules in test set
    inds2 = sorted(range(len(test_prop)), key=lambda k: test_prop[k], reverse=False)
    train_smiles = [train_mols[i] for i in inds2]
    train_scores = [train_prop[i] for i in inds2]
    # print(test_scores[:800])
    
    init_dataset = SimpleDataset(test_smiles[:800], test_scores[:800], transform_fn)
    # init_dataset = SimpleDataset(test_smiles[:3000], test_scores[:3000], transform_fn)
    init_dataloader = DataLoader(init_dataset, batch_size=800, shuffle=False)

    t_end = time.time()

    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(t_end - t_start))
    print('==========================================')
    
    ### Load trained model
    model = GraphEBM(n_atom_type, args.hidden, n_edge_type, args.swish, args.depth, add_self = args.add_self)
    
    print("Loading hyperparamaters from {}".format(args.model_dir))
    model.load_state_dict(torch.load(args.model_dir))
    model = model.to(device)
    

    t_end = time.time()

    print('Load trained model done! Time {:.2f} seconds'.format(t_end - t_start))
    print('==========================================')



#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
    gen_time=[]
    total_optim_dict = dict()
    ### Random generation
    t_start = time.time()
    for run in tqdm(range(args.runs)):
        
        optim_dict_one_try, save_dict = generate(model, init_dataloader, n_atom, n_atom_type, n_edge_type, device, atomic_num_list)
        update_total_optim_dict(total_optim_dict, optim_dict_one_try)
    
    
    t_end = time.time()
    optim_dict = total_optim_dict
    all_keys = list(optim_dict.keys())
    imp0 = [optim_dict[i][0][1] for i in all_keys if optim_dict[i][0][1] != -100]
    imp2 = [optim_dict[i][1][1] for i in all_keys if optim_dict[i][1][1] != -100]
    imp4 = [optim_dict[i][2][1] for i in all_keys if optim_dict[i][2][1] != -100]
    imp6 = [optim_dict[i][3][1] for i in all_keys if optim_dict[i][3][1] != -100]
    sim0 = [optim_dict[i][0][2] for i in all_keys if optim_dict[i][0][1] != -100]
    sim2 = [optim_dict[i][1][2] for i in all_keys if optim_dict[i][1][1] != -100]
    sim4 = [optim_dict[i][2][2] for i in all_keys if optim_dict[i][2][1] != -100]
    sim6 = [optim_dict[i][3][2] for i in all_keys if optim_dict[i][3][1] != -100]
    suc0 = [1 for i in all_keys if optim_dict[i][0][1] != -100]
    suc2 = [1 for i in all_keys if optim_dict[i][1][1] != -100]
    suc4 = [1 for i in all_keys if optim_dict[i][2][1] != -100]
    suc6 = [1 for i in all_keys if optim_dict[i][3][1] != -100]
    print(args.exp_name + '\n')
    print('generation done! Time {:.2f} seconds'.format(t_end - t_start))
    print('Sim 0 mean improvement {:.2f} +- {:.2f} Sim 2 mean improvement {:.2f} +- {:.2f} Sim 4 mean improvement {:.2f} +- {:.2f} Sim 6 mean improvement {:.2f} +- {:.2f}'.format(np.mean(imp0), np.std(imp0), np.mean(imp2), np.std(imp2), np.mean(imp4), np.std(imp4), np.mean(imp6), np.std(imp6)))
    print('Sim 0 mean similarity {:.2f} +- {:.2f} Sim 2 mean similarity {:.2f} +- {:.2f} Sim 4 mean similarity {:.2f} +- {:.2f} Sim 6 mean similarity {:.2f} +- {:.2f}'.format(np.mean(sim0), np.std(sim0), np.mean(sim2), np.std(sim2), np.mean(sim4), np.std(sim4), np.mean(sim6), np.std(sim6)))
    print('Sim 0 success {:.2f} Sim 2 success {:.2f} Sim 4 success {:.2f} Sim 6 success {:.2f}'.format(np.sum(suc0)/800, np.sum(suc2)/800, np.sum(suc4)/800, np.sum(suc6)/800))
    # np.save(args.exp_name + '.npy', save_dict)
    

        

    


            

    