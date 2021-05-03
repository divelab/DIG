import os
from mol_tree import MolTree
import pickle
import math, random, sys
from optparse import OptionParser
from multiprocessing import Pool

from vocab import Vocab
from jtnn_vae import JTNNVAE
from datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset

import numpy as np

import rdkit
from rdkit import RDLogger
from mol_tree import MolTree

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dig.ggraph.method import Generator

class JTVAE(Generator):
    r"""
    The method class for the JTVAE algorithm proposed in the paper `Junction Tree Variational Autoencoder for Molecular Graph Generation <https://arxiv.org/abs/1802.04364>`_. This class provides interfaces for running random generation with the JTVAE algorithm. Please refer to the `benchmark codes <https://github.com/divelab/DIG/tree/dig/benchmarks/ggraph/JTVAE>`_ for usage examples.

    Args:
        training (boolean): If we are training (as opposed to testing).
        build_vocab (boolean): If we need to build the vocabulary (first time training with this dataset).
        device (torch.device, optional): The device where the model is deployed.
    """
    def __init__(self, list_smiles, training=True, build_vocab=True, device=None):
        #super().__init__()
        self.vocab = self.build_vocabulary(list_smiles)
        self.vae = JTNNVAE(Vocab(self.vocab), 450, 56, 20, 3).cuda()
        
    def build_vocabulary(self, list_smiles):
        r"""
            Building the vocabulary for training.
            Args:
                dataset (list): the list of smiles strings in the dataset.
            :rtype:
                cset (list): A list of smiles that contains the vocabulary for the training data.       
        """
        cset = set()
        for smiles in list_smiles:
            mol = MolTree(smiles)
            for c in mol.nodes:
                cset.add(c.smiles)
       # cset_newline = list(map(lambda x: x + "\n", cset))
        return list(cset)
            
    def _tensorize(self, smiles, assm=True):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        return mol_tree 
    
    def preprocess(self, list_smiles):
        r"""
            Preprocess the molecules.
            Args:
                list_smiles (list): The list of smiles strings in the dataset.
            :rtype:
            preprocessed (list): A list of preprocessed MolTree objects.
                
        """
        preprocessed = list(map(self._tensorize, tqdm(list_smiles, leave=True)))
        return preprocessed


    def train_rand_gen(self, loader, load_epoch, lr, anneal_rate, clip_norm, num_epochs, beta, max_beta, step_beta, anneal_iter, kl_anneal_iter, print_iter, save_iter):
        r"""
            Train the Junction Tree Variational Autoencoder.
            Args:
                loader (MolTreeFolder): The MolTreeFolder loader.
                load_epoch (int): The epoch to load from state dictionary.
                lr (float): The learning rate for training.
                anneal_rate (float): The learning rate annealing.
                clip_norm (float): Clips gradient norm of an iterable of parameters.
                num_epochs (int): The number of training epochs.
                beta (float): The KL regularization weight.
                max_beta (float): The maximum KL regularization weight.
                step_beta (float): The KL regularization weight step size.
                anneal_iter (int): How often to step in annealing the learning rate.
                kl_anneal_iter (int): How often to step in annealing the KL regularization weight.
                print_iter (int): How often to print the iteration statistics.
                save_iter (int): How often to save the iteration statistics.

        """
        vocab = Vocab(self.vocab)

        for param in self.vae.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if 0 > 0:
            self.vae.load_state_dict(torch.load('checkpoints' + "/model.iter-" + str(load_epoch)))

        print("Model #Params: %dK" % (sum([x.nelement() for x in self.vae.parameters()]) / 1000,))

        optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
        scheduler.step()

        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

        total_step = load_epoch
        meters = np.zeros(4)

        for epoch in range(num_epochs):
            for batch in loader:
                total_step += 1
                try:
                    self.vae.zero_grad()
                    loss, kl_div, wacc, tacc, sacc = self.vae(batch, beta)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.vae.parameters(), clip_norm)
                    optimizer.step()
                except Exception as e:
                    print(e)
                    continue

                meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

                if total_step % print_iter == 0:
                    meters /= 50
                    print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(self.vae), grad_norm(self.vae)))
                    sys.stdout.flush()
                    meters *= 0

                if total_step % save_iter == 0:
                    torch.save(self.vae.state_dict(), "saved" + "/model.iter-" + str(total_step))

                if total_step % anneal_iter == 0:
                    scheduler.step()
                    print("learning rate: %.6f" % scheduler.get_lr()[0])

                if total_step % kl_anneal_iter == 0 and total_step >= anneal_iter:
                    beta = min(max_beta, beta + step_beta)
                    
    def run_rand_gen(self, num_samples):
        r"""
        Sample new molecules from the trained model.
        Args:
            num_samples (int): Number of samples to generate from the trained model.
        :rtype:
            samples (list): samples is a list of generated molecules.
                
        """
        torch.manual_seed(0)
        samples = [self.vae.sample_prior() for _ in range(num_samples)]
        return samples