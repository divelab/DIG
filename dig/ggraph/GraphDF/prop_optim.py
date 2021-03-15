import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import GraphFlowModel_rl
from rdkit import Chem
import os
import sys
sys.path.append('..')
from utils import metric_property_optimization


def adjust_learning_rate(optimizer, cur_iter, init_lr, warm_up_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if warm up step is 0, no warm up actually.
    if cur_iter < warm_up_step:
        lr = init_lr * (1. / warm_up_step + 1. / warm_up_step * cur_iter)  # [0.1lr, 0.2lr, 0.3lr, ..... 1lr]
    else:
        lr = init_lr
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PropOptim(object):
    def __init__(self, config, out_path=None):
        self.config = config
        self.out_path = out_path
        
        self.model = GraphFlowModel_rl(self.config['rl'], self.config['net'], out_path)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), **self.config['optim'])

        self.start_iter = 0
        if self.config['ckpt_file'] is not None:
            self.load_ckpt(self.config['ckpt_file'])
        elif self.config['pretrain_model'] is not None:
            self.load_pretrain_model(self.config['pretrain_model'])
    

    def load_pretrain_model(self, path):
        load_key = torch.load(path)
        for key in load_key.keys():
            if key in self.model.state_dict().keys():
                self.model.state_dict()[key].copy_(load_key[key].detach().clone())


    def save_ckpt(self, iter):
        net_dict = self.model.state_dict()
        checkpoint = {
            "net": net_dict,
            'optimizer': self.optimizer.state_dict(),
            "iter": iter,
        }
        torch.save(checkpoint, os.path.join(self.out_path, '{}_ckpt.pth'.format(iter)))


    def load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_iter = checkpoint['iter'] + 1
    

    def generate_molecule(self, num=100, num_min_node=0, num_max_node=48, temperature=[0.3, 0.3], save_mols=False, verbose=False):
        self.model.eval()
        all_mols, all_smiles = [], []
        cnt_mol = 0

        while cnt_mol < num:
            mol, num_atoms = self.model.reinforce_optim_one_mol(atom_list=self.config['data']['atom_list'], max_size_rl=num_max_node, temperature=temperature)
            if mol is not None:
                smile = Chem.MolToSmiles(mol)
                if num_atoms >= num_min_node and not smile in all_smiles:
                    all_mols.append(mol)
                    all_smiles.append(smile)
                    cnt_mol += 1
                    if verbose and cnt_mol % 10 == 0:
                        print('Generated {} molecules'.format(cnt_mol))

        top3 = metric_property_optimization(all_mols, topk=3, prop=self.config['rl']['property_type'])

        if save_mols and self.out_path is not None:
            f = open(os.path.join(self.out_path, 'smiles.txt'), 'w')
            for i in range(3):
                f.write('{} {}\n'.format(top3[i][0], top3[i][1]))
            f.close()
            
        return top3

    
    def reinforce_one_iter(self, iter_cnt, in_baseline=None):
        self.optimizer.zero_grad()    
        loss, per_mol_reward, per_mol_property_score, out_baseline = self.model.reinforce_forward_optim(
                                                atom_list=self.config['data']['atom_list'], temperature=self.config['temperature_for_gen'], max_size_rl=self.config['num_max_node_for_gen'],  
                                                batch_size=self.config['batch_size'], in_baseline=in_baseline, cur_iter=iter_cnt)

        num_mol = len(per_mol_reward)
        avg_reward = sum(per_mol_reward) / num_mol
        avg_score = sum(per_mol_property_score) / num_mol     
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), 1.0)
        adjust_learning_rate(self.optimizer, iter_cnt, self.config['optim']['lr'], self.config['rl']['warm_up'])
        self.optimizer.step()
  
        return loss.item(), avg_reward, avg_score, out_baseline
    

    def reinforce(self):
        moving_baseline = None
        print('start finetuning model(reinforce)')
        for cur_iter in range(self.start_iter, self.config['reinforce_iters']):
            if cur_iter == 0:
                iter_loss, iter_reward, iter_score, moving_baseline = self.reinforce_one_iter(cur_iter, in_baseline=None)
            else:
                iter_loss, iter_reward, iter_score, moving_baseline = self.reinforce_one_iter(cur_iter, in_baseline=moving_baseline)
            print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, iter_reward, iter_score, iter_loss))

            if self.out_path is not None:               
                f = open(os.path.join(self.out_path, 'record.txt'), 'a')
                f.write('Iter {} | reward {}, score {}, loss {}\n'.format(cur_iter, iter_reward, iter_score, iter_loss))
                f.close()
                if cur_iter % self.config['save_iters'] == self.config['save_iters'] - 1:
                    torch.save(self.model.state_dict(), os.path.join(self.out_path, 'prop_optim_net_{}.pth'.format(cur_iter)))

        print("Finetuning(Reinforce) Finished!")