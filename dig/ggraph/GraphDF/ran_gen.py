import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.serialization import save
from torch.utils.data import DataLoader
from rdkit import Chem
from model import GraphFlowModel
import sys
sys.path.append('..')
from utils import MolSet, metric_random_generation



class RandomGen(object):
    def __init__(self, config, smile_list, out_path=None):
        self.config = config
        self.out_path = out_path

        num_max_node, num_bond_type, atom_list, use_aug = self.config['data']['num_max_node'], self.config['data']['num_bond_type'], self.config['data']['atom_list'], self.config['data']['use_aug']
        self.dataset = MolSet(smile_list=smile_list, num_max_node=num_max_node, num_bond_type=num_bond_type, edge_unroll=self.config['net']['edge_unroll'], atom_list=atom_list, use_aug=use_aug)
        self.dataloader = DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=True)
        self.model = GraphFlowModel(**self.config['net'])
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **self.config['optim'])

        self.start_epoch = 1
        if self.config['ckpt_file'] is not None:
            self.load_ckpt(self.config['ckpt_file'])


    def save_ckpt(self, epoch):
        net_dict = self.model.state_dict()
        checkpoint = {
            "net": net_dict,
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(self.out_path, '{}_ckpt.pth'.format(epoch)))


    def load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1


    def _train_epoch(self):
        total_loss = 0
        self.model.train()
        for batch, data_batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            inp_node_features = data_batch['node'] #(B, N, node_dim)
            inp_adj_features = data_batch['adj'] #(B, 4, N, N)
            if self.config['net']['use_gpu']:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()
            
            out_z = self.model(inp_node_features, inp_adj_features)
            loss = self.model.dis_log_prob(out_z)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.to('cpu').item()
            if batch % self.config['verbose'] == 0:
                print('Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

        return total_loss / (batch + 1)


    def train(self):      
        epoches = self.config['epoches']
        for epoch in range(self.start_epoch, epoches+1):
            avg_loss = self._train_epoch()
            print("Training | Average loss {}".format(avg_loss))
            
            if epoch % self.config['save_epoch'] == 0:
                unique_rate, valid_no_check_rate, valid_rate, novelty_rate, _ = self.evaluate(num=self.config['num_gen'], 
                    num_min_node=self.config['num_min_node_for_gen'], num_max_node=self.config['num_max_node_for_gen'], temperature=self.config['temperature_for_gen'])
                print("Validation | unique rate {} valid rate no check {} valid rate {} novelty rate {}".format(unique_rate, valid_no_check_rate, valid_rate, novelty_rate))

                if self.out_path is not None:
                    file_obj = open(os.path.join(self.out_path, 'record.txt'), 'a')
                    file_obj.write('unique rate {} valid rate no check {} valid rate {} novelty rate {}\n'.format(unique_rate, valid_no_check_rate, valid_rate, novelty_rate))
                    file_obj.write('Average loss {}\n'.format(avg_loss))
                    file_obj.close()
                    torch.save(self.model.state_dict(), os.path.join(self.out_path, 'ran_gen_net_{}.pth'.format(epoch)))


    def evaluate(self, num=100, num_min_node=7, num_max_node=20, temperature=[0.3, 0.3], save_mols=False, verbose=False):
        pure_valids, all_mols = self.generate_molecule(num=num, num_min_node=num_min_node, num_max_node=num_max_node, temperature=temperature, verbose=verbose)
        metrics = metric_random_generation(all_mols, self.dataset.all_smiles)
        unique_rate, valid_rate, novelty_rate = metrics['unique_ratio'], metrics['valid_ratio'], metrics['novel_ratio']
        valid_no_check_rate = sum(pure_valids) / num
        print("Valid Ratio without valency check: {}/{} = {:.2f}".format(int(sum(pure_valids)), num, valid_no_check_rate*100))

        if save_mols and self.out_path is not None:
            file_obj = open(os.path.join(self.out_path, 'smiles.txt'), 'w')
            for mol in all_mols:
                smiles = Chem.MolToSmiles(mol)
                file_obj.write(smiles+'\n')
            file_obj.close()

        return unique_rate, valid_no_check_rate*100, valid_rate, novelty_rate, all_mols


    def generate_molecule(self, num=100, num_min_node=7, num_max_node=25, temperature=[0.3, 0.3], verbose=False):
        self.model.eval()
        all_mols, pure_valids = [], []
        cnt_mol = 0

        while cnt_mol < num:
            mol, no_resample, num_atoms = self.model.generate(atom_list=self.config['data']['atom_list'], min_atoms=num_min_node, max_atoms=num_max_node, temperature=temperature)
            if (num_atoms >= num_min_node):
                cnt_mol += 1
                all_mols.append(mol)
                pure_valids.append(no_resample)
                if verbose and cnt_mol % 10 == 0:
                    print('Generated {} molecules'.format(cnt_mol))

        assert cnt_mol == num, 'number of generated molecules does not equal num'        

        return pure_valids, all_mols