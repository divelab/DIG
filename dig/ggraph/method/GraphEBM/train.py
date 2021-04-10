import argparse
import time
import sys
import os
from tqdm import tqdm
import random

import torch
from torch.optim import Adam
from texttable import Texttable
from torch.utils.data import DataLoader, random_split, Subset
from distutils.util import strtobool
import numpy as np


from preprocess_data import transform_qm9, transform_zinc250k
from model import *
from preprocess_data.data_loader import NumpyTupleDataset
from util import *


### Args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
parser.add_argument('--data_dir', type=str, default='./preprocess_data', help='Location for the dataset')
parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='Dataset name')
parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
parser.add_argument('--depth', type=int, default=2, help='Number of graph conv layers')
parser.add_argument('--add_self', type=strtobool, default='false', help='Add shortcut in graphconv')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
parser.add_argument('--swish', type=strtobool, default='true', help='Use swish as activation function')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
parser.add_argument('--shuffle', type=strtobool, default='true', help='Shuffle the data batch')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers in the data loader')
parser.add_argument('--c', type=float, default=0.5, help='Dequantization using uniform distribution of [0,c)')
parser.add_argument('--alpha', type=float, default=1.0, help='Weight for energy magnitudes regularizer')
parser.add_argument('--step_size', type=int, default=10, help='Step size in Langevin dynamics')
parser.add_argument('--sample_step', type=int, default=30, help='Number of sample step in Langevin dynamics')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--noise', type=float, default=0.005, help='The standard variance of the added noise during Langevin Dynamics')
parser.add_argument('--clamp', type=strtobool, default='true', help='Clamp the data/gradient during Langevin Dynamics')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay')
parser.add_argument('--max_epochs', type=int, default=20, help='Maximum training epochs')
parser.add_argument('--save_dir', type=str, default='trained_models/qm9', help='Location for saving checkpoints')
parser.add_argument('--save_interval', type=int, default=1, help='Interval (# of epochs) between saved checkpoints')
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

                
  
        
def train(model, train_dataloader, n_atom, n_atom_type, n_edge_type, device):
    parameters = model.parameters()
    optimizer = Adam(parameters, lr=args.lr, betas=(0.0, 0.999), weight_decay=args.wd)
    for epoch in range(args.max_epochs):
        t_start = time.time()
        losses_reg = []
        losses_en = []
        losses = []
        for i, batch in enumerate(tqdm(train_dataloader)):
            ### Dequantization
            pos_x = batch[0].to(device) 
            pos_x += args.c * torch.rand_like(pos_x, device=device)  # (128, 9, 5)
            pos_adj = batch[1].to(device) 
            pos_adj += args.c * torch.rand_like(pos_adj, device=device)  # (128, 4, 9, 9)

            
            ### Langevin dynamics
            neg_x = torch.rand(pos_x.shape[0], n_atom, n_atom_type, device=device) * (1 + args.c) # (128, 9, 5)
            neg_adj = torch.rand(pos_adj.shape[0], n_edge_type, n_atom, n_atom, device=device) #(128, 4, 9, 9)
        
            pos_adj = rescale_adj(pos_adj)
            neg_x.requires_grad = True
            neg_adj.requires_grad = True
            
            
            
            requires_grad(parameters, False)
            model.eval()
            

            
            noise_x = torch.randn(neg_x.shape[0], n_atom, n_atom_type, device=device)  # (128, 9, 5)
            noise_adj = torch.randn(neg_adj.shape[0], n_edge_type, n_atom, n_atom, device=device)  #(128, 4, 9, 9) 
            for k in range(args.sample_step):

                noise_x.normal_(0, args.noise)
                noise_adj.normal_(0, args.noise)
                neg_x.data.add_(noise_x.data)
                neg_adj.data.add_(noise_adj.data)

                neg_out = model(neg_adj, neg_x)
                neg_out.sum().backward()
                if args.clamp:
                    neg_x.grad.data.clamp_(-0.01, 0.01)
                    neg_adj.grad.data.clamp_(-0.01, 0.01)
        

                neg_x.data.add_(neg_x.grad.data, alpha=-args.step_size)
                neg_adj.data.add_(neg_adj.grad.data, alpha=-args.step_size)

                neg_x.grad.detach_()
                neg_x.grad.zero_()
                neg_adj.grad.detach_()
                neg_adj.grad.zero_()
                
                neg_x.data.clamp_(0, 1 + args.c)
                neg_adj.data.clamp_(0, 1)

            ### Training by backprop
            neg_x = neg_x.detach()
            neg_adj = neg_adj.detach()
            requires_grad(parameters, True)
            model.train()
            
            model.zero_grad()
            
            pos_out = model(pos_adj, pos_x)
            neg_out = model(neg_adj, neg_x)
            
            loss_reg = (pos_out ** 2 + neg_out ** 2)  # energy magnitudes regularizer
            loss_en = pos_out - neg_out  # loss for shaping energy function
            loss = loss_en + args.alpha * loss_reg
            loss = loss.mean()
            loss.backward()
            clip_grad(parameters, optimizer)
            optimizer.step()
            

            losses_reg.append(loss_reg.mean())
            losses_en.append(loss_en.mean())
            losses.append(loss)
            
            
        t_end = time.time()
        
        ### Save checkpoints
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}.pt'.format(epoch + 1)))
            print('Saving checkpoint at epoch ', epoch+1)
            print('==========================================')
        print('Epoch: {:03d}, Loss: {:.6f}, Energy Loss: {:.6f}, Regularizer Loss: {:.6f}, Sec/Epoch: {:.2f}'.format(epoch+1, (sum(losses)/len(losses)).item(), (sum(losses_en)/len(losses_en)).item(), (sum(losses_reg)/len(losses_reg)).item(), t_end-t_start))
        print('==========================================')




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Load dataset
    t_start = time.time()

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


    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, data_file), transform=transform_fn) # 133885
    if len(valid_idx) > 0:
        train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
        train_set = Subset(dataset, train_idx)  # 120,803
        test_set = Subset(dataset, valid_idx)  # 13,082
    else:
        torch.manual_seed(args.seed)
        train_set, test_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    t_end = time.time()

    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(t_end - t_start))
    print('==========================================')


    ### Initialize model
    model = GraphEBM(n_atom_type, args.hidden, n_edge_type, args.swish, args.depth, add_self=args.add_self, dropout = args.dropout)
    print(model)
    print('==========================================')
    model = model.to(device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    ### Train
    train(model, train_dataloader, n_atom, n_atom_type, n_edge_type, device)
    
    


            

    