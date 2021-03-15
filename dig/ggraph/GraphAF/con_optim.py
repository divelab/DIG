import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from rdkit import Chem
import os
import sys
sys.path.append('..')
from utils import MolSet, get_smiles_props_800, metric_constrained_optimization


class DataIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        
    def __next__(self):
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


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


class ConOptim(object):
    def __init__(self, config, data_file, out_path=None):
        self.config = config
        self.out_path = out_path

        num_max_node, num_bond_type, atom_list, use_aug = self.config['data']['num_max_node'], self.config['data']['num_bond_type'], self.config['data']['atom_list'], self.config['data']['use_aug']
        smile_list, prop_list = get_smiles_props_800(data_file)
        self.data_file = data_file
        self.dataset = MolSet(smile_list=smile_list, prop_list=prop_list, num_max_node=num_max_node, num_bond_type=num_bond_type, edge_unroll=self.config['net']['edge_unroll'], atom_list=atom_list, use_aug=use_aug)
        self.dataloader = DataIterator(DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=True))
        self.model = GraphFlowModel_con_rl(self.config['rl'], self.config['net'])
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
            "iter": iter
        }
        torch.save(checkpoint, os.path.join(self.out_path, '{}_ckpt.pth'.format(iter)))


    def load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_iter = checkpoint['iter'] + 1
    
    def load_pth(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
        
    def constrained_optimize_one_mol_rl(self, adj, x, org_smile, mol_size, bfs_perm_origin, max_size_rl=38, temperature=0.3):
        """
        direction: score ascent direction
        adj: adjacent matrix of origin mol (1, 4, N, N)
        x: node feature of origin mol (1, N, 9)
        """
        self.model.eval()

        best_mol0 = None
        best_mol2 = None
        best_mol4 = None
        best_mol6 = None
        best_imp0 = -100.
        best_imp2 = -100.
        best_imp4 = -100.
        best_imp6 = -100.
        final_sim0 = -1.
        final_sim2 = -1.
        final_sim4 = -1.
        final_sim6 = -1.

        mol_org = Chem.MolFromSmiles(org_smile)
        mol_org_size = mol_org.GetNumAtoms()
        assert mol_org_size == mol_size

        cur_mols, cur_mol_imps, cur_mol_sims = self.model.reinforce_constrained_optim_one_mol(x, adj, mol_size, org_smile, bfs_perm_origin,
                                                                        self.config['data']['atom_list'], temperature=temperature, max_size_rl=max_size_rl)
        num_success = len(cur_mol_imps)
        for i in range(num_success):
            cur_mol = cur_mols[i]
            cur_imp = cur_mol_imps[i]
            cur_sim = cur_mol_sims[i]
            assert cur_imp > 0
            if cur_sim > 0:
                if cur_imp > best_imp0:
                    best_mol0 = cur_mol
                    best_imp0 = cur_imp
                    final_sim0 = cur_sim
            if cur_sim > 0.2:
                if cur_imp > best_imp2:
                    best_mol2 = cur_mol
                    best_imp2 = cur_imp
                    final_sim2 = cur_sim
            if cur_sim > 0.4:
                if cur_imp > best_imp4:
                    best_mol4 = cur_mol
                    best_imp4 = cur_imp
                    final_sim4 = cur_sim
            if cur_sim > 0.6:
                if cur_imp > best_imp6:
                    best_mol6 = cur_mol
                    best_imp6 = cur_imp
                    final_sim6 = cur_sim                    

        return [best_mol0, best_mol2, best_mol4, best_mol6], [best_imp0, best_imp2, best_imp4, best_imp6], [final_sim0, final_sim2, final_sim4, final_sim6]
    

    def constrained_optimize(self, num_max_node=38, temperature=0.3, save_mol=True):
        data_len = len(self.dataset)
        repeat_time, min_optim_time = self.config['rl']['repeat_time'], self.config['rl']['min_optim_time']
        
        optim_success_dict = {}
        mols_0, mols_2, mols_4, mols_6 = [], [], [], []
        for batch_cnt in range(data_len):
            best_mol = [None, None, None, None]
            best_score = [-100., -100., -100., -100.]
            final_sim = [-1., -1., -1., -1.]

            batch_data = self.dataset[batch_cnt] # dataloader is dataset object

            inp_node_features = batch_data['node'].unsqueeze(0) #(1, N, node_dim)              
            inp_adj_features = batch_data['adj'].unsqueeze(0) #(1, 4, N, N)              

            raw_smile = batch_data['raw_smile']  #(1)
            mol_size = batch_data['mol_size']
            bfs_perm_origin = batch_data['bfs_perm_origin']
            bfs_perm_origin = torch.Tensor(bfs_perm_origin)

            for cur_iter in range(repeat_time):
                if raw_smile not in optim_success_dict:
                    optim_success_dict[raw_smile] = [0, -1] #(try_time, imp)
                if optim_success_dict[raw_smile][0] > min_optim_time and optim_success_dict[raw_smile][1] > 0: # reach min time and imp is positive
                    continue # not optimize this one

                best_mol0246, best_score0246, final_sim0246 = self.constrained_optimize_one_mol_rl(inp_adj_features, 
                                                                    inp_node_features, raw_smile, mol_size, bfs_perm_origin, num_max_node, temperature)
                if best_score0246[0] > best_score[0]:
                    best_score[0] = best_score0246[0]
                    best_mol[0] = best_mol0246[0]
                    final_sim[0] = final_sim0246[0]

                if best_score0246[1] > best_score[1]:
                    best_score[1] = best_score0246[1]
                    best_mol[1] = best_mol0246[1]
                    final_sim[1] = final_sim0246[1] 

                if best_score0246[2] > best_score[2]:
                    best_score[2] = best_score0246[2]
                    best_mol[2] = best_mol0246[2]
                    final_sim[2] = final_sim0246[2]
                    
                if best_score0246[3] > best_score[3]:
                    best_score[3] = best_score0246[3]
                    best_mol[3] = best_mol0246[3]
                    final_sim[3] = final_sim0246[3]

                if best_score[3] > 0: #imp > 0
                    optim_success_dict[raw_smile][1] = best_score[3]
                optim_success_dict[raw_smile][0] += 1 # try time + 1

            mols_0.append(best_mol[0])
            mols_2.append(best_mol[1])
            mols_4.append(best_mol[2])
            mols_6.append(best_mol[3])
            
            if save_mol and self.out_path is not None:
                best_smile = [None, None, None, None]
                for i,mol in enumerate(best_mol):
                    if mol is not None:
                        best_smile[i] = Chem.MolToSmiles(mol, isomericSmiles=True)
                fp = open(os.path.join(self.out_path, 'optim.txt'), 'a')
                fp.write('{}\n'.format(raw_smile))
                fp.write('{}\n'.format(best_smile))
                fp.write('{}\n'.format(best_score))
                fp.write('{}\n'.format(final_sim))
                fp.close()

            if batch_cnt % 1 == 0:
                print('Optimized {} molecules'.format(batch_cnt+1))
        
        metrics = metric_constrained_optimization(mols_0, mols_2, mols_4, mols_6, self.data_file)

        return metrics


    def reinforce_one_iter(self, iter_cnt, in_baseline=None):
        self.optimizer.zero_grad()
        batch_data = next(self.dataloader)
        mol_xs = batch_data['node']
        mol_adjs = batch_data['adj']
        mol_sizes = batch_data['mol_size']
        bfs_perm_origin = batch_data['bfs_perm_origin']
        raw_smiles = batch_data['raw_smile']
    
        loss, per_mol_reward, per_mol_property_score, out_baseline = self.model.reinforce_forward_constrained_optim(
                                                mol_xs=mol_xs, mol_adjs=mol_adjs, mol_sizes=mol_sizes, raw_smiles=raw_smiles, bfs_perm_origin=bfs_perm_origin, atom_list=self.config['data']['atom_list'],
                                                temperature=self.config['temperature_for_gen'],  max_size_rl=self.config['num_max_node_for_gen'],  
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
                    torch.save(self.model.state_dict(), self.out_path+'con_optim_net_{}.pth'.format(cur_iter))

        print("Finetuning(Reinforce) Finished!")