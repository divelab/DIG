import os
from mol_tree import MolTree
import pickle

from vocab import Vocab
from jtnn_vae import JTNNVAE
from datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset

from multiprocessing import Pool

import os
import math, random, sys
from optparse import OptionParser
import pickle

import numpy as np

#from fast_jtnn import *
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


#from dig.ggraph.method import Generator

class JTVAE(object): #TODO: implement child of Generator):
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
        
    def build_vocabulary(self, list_smiles: str):
        r"""
            Building the vocabulary for training.
            Args:
                dataset (str): the path to the dataset.
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
        """
        num_splits = 1  # TODO Reassign
#         if not os.path.exists("moses-preprocessed"):
#             os.mkdir("moses-preprocessed")
#         with open(os.path.join("datasets", "moses.csv"), "r") as f:
#             data = [line.strip("\r\n ").split()[0] for line in f]

#         pool = Pool(8)  # TODO arg
#         all_data = pool.map(self._tensorize, list_smiles)

#         le = (len(all_data) + num_splits - 1) // num_splits

#         for split_id in range(num_splits):
#             st = split_id * le
#             sub_data = all_data[st : st + le]

#             with open('moses-processed/tensors-%d.pkl' % split_id, 'wb') as f:
#                 pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
        return list(map(self._tensorize, tqdm(list_smiles, leave=True)))


    def train(self, preprocessed):
        r"""
            Train the Junction Tree Variational Autoencoder.
        """
        # Constants TODO remove
        load_epoch = 0
        lr = 1e-3
        anneal_rate = 0.9
        clip_norm = 50.0
        num_epochs = 1
        
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

        total_step = 0  # TODO args.load_epoch
        beta = 0.0  # TODO args.beta
        meters = np.zeros(4)

        for epoch in range(num_epochs):
            loader = MolTreeFolder(preprocessed, vocab, 32, num_workers=4)
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

                if total_step % 50 == 0:  # TODO all save_iters replace
                    meters /= 50
                    print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(self.vae), grad_norm(self.vae)))
                    sys.stdout.flush()
                    meters *= 0

                if total_step % 5000 == 0:
                    torch.save(self.vae.state_dict(), "saved" + "/model.iter-" + str(total_step))

                if total_step % 40000 == 0:
                    scheduler.step()
                    print("learning rate: %.6f" % scheduler.get_lr()[0])

                if total_step % 2000 == 0 and total_step >= 40000:
                    beta = min(1.0, beta + 0.002)
                    
    def sample(self, num_samples):
        r"""
        Sample new molecules from the trained model.
        """
#         vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
#         vocab = Vocab(vocab)

#         model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
#         model.load_state_dict(torch.load(args.model))
#         model = model.cuda()

        torch.manual_seed(0)
#         for i in range(num_samples):
#             print(self.vae.sample_prior())
        return [self.vae.sample_prior() for _ in range(num_samples)]