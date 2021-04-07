import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from data_io_tmp import *
import os
from tmp_utils import *
import time


class PropOptim(object):
    def __init__(self, config, out_path=None):
        self.config = config
        self.out_path = out_path
        
        self.model = GraphFlowModel_rl(self.config['rl'], self.config['net'], out_path)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), **self.config['optim'])

        self.start_iter = 0
        self.best_reward = -100.0
        if self.config['ckpt_file'] is not None:
            self.load_ckpt(self.config['ckpt_file'])
        elif self.config['dense_gen_model'] is not None:
            self.load_dense_gen_model(self.config['dense_gen_model'])
            # self.initialize_from_checkpoint(self.config['dense_gen_model'])
    

    def load_dense_gen_model(self, path):
        load_key = torch.load(path)
        for key in load_key.keys():
            if key in self.model.state_dict().keys():
                self.model.state_dict()[key].copy_(load_key[key].detach().clone())


    def initialize_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        print(checkpoint['model_state_dict'].keys()) # check the key consistency     
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)       

        print('initialize from %s Done!' % path)
    
    def save_ckpt(self, iter):
        net_dict = self.model.state_dict()
        checkpoint = {
            "net": net_dict,
            'optimizer': self.optimizer.state_dict(),
            "iter": iter,
            "best_reward": self.best_reward,
        }
        torch.save(checkpoint, self.out_path+'{}_ckpt.pth'.format(iter))


    def load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_iter = checkpoint['iter'] + 1
        self.best_reward = checkpoint['best_reward']
        
    def load_pth(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
    

    def generate_molecule(self, num=100, num_min_node=0, num_max_node=48, temperature=0.75, verbose=False):
        self.model.eval()
        all_smiles, all_props = [], []
        cnt_mol = 0

        while cnt_mol < num:
            smiles, props, num_atoms = self.model.reinforce_optim_one_mol(atom_list=self.config['data']['atom_list'], max_size_rl=num_max_node, temperature=temperature)
            for i,smile in enumerate(smiles):
                if num_atoms[i] >= num_min_node and not smile in all_smiles:
                    all_smiles.append(smile)
                    all_props.append(props[i])

                    cnt_mol += 1
                    if verbose and cnt_mol % 10 == 0:
                        print('Generated {} molecules'.format(cnt_mol))

        sorted_props_idx = np.argsort(all_props)
        top_3_idx = sorted_props_idx[-3:]
        top_3_smiles = [all_smiles[idx] for idx in top_3_idx]
        top_3_props = [all_props[idx] for idx in top_3_idx]
        mean = np.array(all_props).mean()
        std = np.array(all_props).std()
        return all_smiles, all_props, top_3_smiles, top_3_props, mean, std

    
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
        _, _, top_3_smiles, top_3_props, mean, std = self.generate_molecule(num=self.config['num_gen'], num_max_node=self.config['num_max_node_for_gen'],
        temperature=self.config['temperature_for_gen'])
        print('\t top 3 props {} mean {}'.format(top_3_props, mean))
        print('start finetuning model(reinforce)')
        for cur_iter in range(self.start_iter, self.config['reinforce_iters']):
            if cur_iter == 0:
                iter_loss, iter_reward, iter_score, moving_baseline = self.reinforce_one_iter(cur_iter, in_baseline=None)
            else:
                iter_loss, iter_reward, iter_score, moving_baseline = self.reinforce_one_iter(cur_iter, in_baseline=moving_baseline)
            print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, iter_reward, iter_score, iter_loss))

            if cur_iter % self.config['valid_iters'] == self.config['valid_iters'] - 1:
                _, _, top_3_smiles, top_3_props, mean, std = self.generate_molecule(num=self.config['num_gen'], num_max_node=self.config['num_max_node_for_gen'],
                temperature=self.config['temperature_for_gen'])
                print('\t top 3 props {} mean {}'.format(top_3_props, mean))

                if self.out_path is not None:
                    f = open(os.path.join(self.out_path, 'record.txt'), 'a')
                    f.write('top 3 properties {}\n'.format(top_3_props))
                    f.write('top 3 smiles {}\t{}\t{}\n'.format(top_3_smiles[0], top_3_smiles[1], top_3_smiles[2]))
                    f.close()

                    torch.save(self.model.state_dict(), self.out_path+'prop_optim_net_{}.pth'.format(cur_iter))

        print("Finetuning(Reinforce) Finished!")
    
    def test_generate(self):
        for it in range(10):
            cur_iter = (it+1)*20-1
            self.model.load_state_dict(torch.load(self.out_path+'prop_optim_net_{}.pth'.format(cur_iter)))
            print('start test generate model of iteration %d'%cur_iter)
            _, _, top_3_smiles, top_3_props, mean, std = self.generate_molecule(num=self.config['num_gen'], num_max_node=self.config['num_max_node_for_gen'],
            temperature=self.config['temperature_for_gen'])
            print('\t top 3 props {}'.format(top_3_props))
            print('\t mean props {} std props {}'.format(mean, std))

            if self.out_path is not None:
                f = open(os.path.join(self.out_path, 'record.txt'), 'a')
                f.write('top 3 properties {}\n'.format(top_3_props))
                f.write('top 3 smiles {}\t{}\t{}\n'.format(top_3_smiles[0], top_3_smiles[1], top_3_smiles[2]))
                f.close()

        print("test generate model Finished!")
        
    def test_generate_pthnum(self,num):

        cur_iter = num
        self.model.load_state_dict(torch.load(self.out_path+'prop_optim_net_{}.pth'.format(cur_iter)))
        print('start test generate model of iteration %d'%cur_iter)
        _, _, top_3_smiles, top_3_props, mean, std = self.generate_molecule(num=20000, num_max_node=self.config['num_max_node_for_gen'],
        temperature=1.0)
        print('\t top 3 props {}'.format(top_3_props))
        print('\t mean props {} std props {}'.format(mean, std))

        if self.out_path is not None:
            f = open(os.path.join(self.out_path, 'record.txt'), 'a')
            f.write('top 3 properties {}\n'.format(top_3_props))
            f.write('top 3 smiles {}\t{}\t{}\n'.format(top_3_smiles[0], top_3_smiles[1], top_3_smiles[2]))
            f.close()

    print("test generate model Finished!")