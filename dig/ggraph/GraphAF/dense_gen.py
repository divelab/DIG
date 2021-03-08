import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from data_io_tmp import *
import time
from rdkit import Chem
import sys
sys.path.append('..')
from utils.data_io import get_smiles_zinc250k


class DensityGen(object):
    def __init__(self, config, mol_file, out_path=None):
        self.config = config
        self.out_path = out_path

        num_max_node, num_bond_type, atom_list = self.config['data']['num_max_node'], self.config['data']['num_bond_type'], self.config['data']['atom_list']
        smile_list = get_smiles_zinc250k(mol_file)
        self.dataset = MolSet(mode='pretrain', smile_list=smile_list, num_max_node=num_max_node, num_bond_type=num_bond_type, edge_unroll=self.config['net']['edge_unroll'], atom_list=atom_list)
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
        torch.save(checkpoint, self.out_path+'{}_ckpt.pth'.format(epoch))


    def load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1

    def load_pth(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint, strict=False)


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
            
            if self.config['net']['use_df']:
                out_z = self.model(inp_node_features, inp_adj_features)
                loss = self.model.dis_log_prob(out_z)
            else:
                out_z, out_logdet = self.model(inp_node_features, inp_adj_features)
                loss = self.model.log_prob(out_z, out_logdet)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.to('cpu').item()
            if batch % self.config['verbose'] == 0:
                print('Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

        return total_loss / (batch + 1)


    def train(self):
        if self.out_path is not None:
            file_obj = open(self.out_path+'record.txt', 'a')
        
        epoches = self.config['epoches']
        for epoch in range(self.start_epoch, epoches+1):
            avg_loss = self._train_epoch()
            print("Training | Average loss {}".format(avg_loss))
            if epoch > 4:
                unique_rate, valid_no_check_rate, valid_rate, novelty_rate, all_smiles, priors = self.evaluate(num=self.config['num_gen'], 
                    num_min_node=self.config['num_min_node_for_gen'], num_max_node=self.config['num_max_node_for_gen'], temperature=self.config['temperature_for_gen'])
                print("Validation | unique rate {} valid rate no check {} valid rate {} novelty rate {}".format(unique_rate, valid_no_check_rate, valid_rate, novelty_rate))

                if self.out_path is not None:
                    fp = open(self.out_path+'dense_gen_{}.txt'.format(epoch), 'w')
                    cnt = 0
                    for i in range(len(all_smiles)):
                        fp.write(all_smiles[i] + '\n')
                        cnt += 1
                    fp.close()
                    print('writing %d smiles into %s done!' % (cnt, self.out_path))

                    file_obj = open(self.out_path+'record.txt', 'a')
                    file_obj.write('unique rate {} valid rate no check {} valid rate {} novelty rate {}\n'.format(unique_rate, valid_no_check_rate, valid_rate, novelty_rate))
                    file_obj.close()
            torch.save(self.model.state_dict(), self.out_path+'dense_gen_net_{}.pth'.format(epoch))


    def evaluate(self, num=100, num_min_node=5, num_max_node=48, temperature=0.75, verbose=False):
        pure_valids, final_valids, all_smiles, priors = self.generate_molecule(num=num, 
            num_min_node=num_min_node, num_max_node=num_max_node, temperature=temperature, verbose=verbose)

        unique_smiles = list(set(all_smiles))
        unique_rate = len(unique_smiles) / num
        valid_no_check_rate = sum(pure_valids) / num
        valid_rate = sum(final_valids) / num

        appear_in_train = 0
        for smiles in all_smiles:
            if self.dataset.all_smiles is not None and smiles in self.dataset.all_smiles:
                    appear_in_train += 1.0
        novelty_rate = 1. - (appear_in_train / num)

        return unique_rate, valid_no_check_rate, valid_rate, novelty_rate, all_smiles, priors


    def generate_molecule(self, num=100, num_min_node=5, num_max_node=48, temperature=0.75, verbose=False):
        self.model.eval()
        all_smiles, pure_valids, final_valids, priors = [], [], [], []
        cnt_mol = 0

        while cnt_mol < num:
            smiles, no_resample, final_valid, num_atoms, prior_latent_nodes =  self.model.generate(atom_list=self.config['data']['atom_list'], max_atoms=num_max_node, temperature=temperature)
            if (num_atoms >= num_min_node):
                cnt_mol += 1
                all_smiles.append(smiles)
                pure_valids.append(no_resample)
                final_valids.append(final_valid)
                priors.append(prior_latent_nodes)
                if verbose and cnt_mol % 10 == 0:
                    print('Generated {} molecules'.format(cnt_mol))

        assert cnt_mol == num, 'number of generated molecules does not equal num'        

        return pure_valids, final_valids, all_smiles, priors


    def reconstruct_molecule(self, smile_list, batch_size=1):
        self.model.eval()
        num_max_node, num_bond_type, atom_list = self.config['data']['num_max_node'], self.config['data']['num_bond_type'], self.config['data']['atom_list']
        dataset = MolSet(mode='pretrain', smile_list=smile_list, num_max_node=num_max_node, num_bond_type=num_bond_type, atom_list=atom_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        recon_list = []
        for batch, data_batch in enumerate(dataloader):
            inp_node_features = data_batch['node'] #(B, N, node_dim)
            inp_adj_features = data_batch['adj'] #(B, 4, N, N)            
            if self.config['net']['use_gpu']:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()
            
            recons = self.model.reconstruct(self.config['data']['atom_list'], inp_node_features, inp_adj_features)
            recon_list.extend(recons)
        
        recon_cnt = 0
        for i in range(len(smile_list)):
            new_mol = Chem.MolFromSmiles(smile_list[i].replace('@','').replace('/', ''))
            new_mol = convert_radical_electrons_to_hydrogens(new_mol)
            new_smile = Chem.MolToSmiles(new_mol)
            if recon_list[i] == new_smile:
                recon_cnt += 1
            else:
                print(recon_list[i], new_smile)
        
        return recon_cnt / len(smile_list)