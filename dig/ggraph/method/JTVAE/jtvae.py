import math
import sys
from optparse import OptionParser

import numpy as np
import rdkit
from rdkit import RDLogger, Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dig.ggraph.method import Generator

from . import fast_jtnn


class JTVAE(Generator):
    r"""
    The method class for the JTVAE algorithm proposed in the paper `Junction Tree Variational Autoencoder for Molecular Graph Generation <https://arxiv.org/abs/1802.04364>`_. This class provides interfaces for running random generation with the JTVAE algorithm. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/JTVAE>`_ for usage examples.

    Args:
        list_smiles (list): List of smiles in training data.
        training (boolean): If we are training (as opposed to testing).
        build_vocab (boolean): If we need to build the vocabulary (first time training with this dataset).
        device (torch.device, optional): The device where the model is deployed.
    """

    def __init__(self, list_smiles, build_vocab=True, device=None):
        super().__init__()
        if build_vocab:
            self.vocab = self.build_vocabulary(list_smiles)
        self.model = None

    def get_model(self, task, config_dict):
        if task == 'rand_gen':
            #hidden_size, latent_size, depthT, depthG
            self.vae = fast_jtnn.JTNNVAE(
                fast_jtnn.Vocab(self.vocab), **config_dict).cuda()
        elif task == 'cons_optim':
            self.prop_vae = jtnn.JTPropVAE(
                jtnn.Vocab(self.vocab), **config_dict).cuda()
        else:
            raise ValueError('Task {} is not supported in JTVAE!'.format(task))

    def build_vocabulary(self, list_smiles):
        r"""
            Building the vocabulary for training.

            Args:
                list_smiles (list): the list of smiles strings in the dataset.

            :rtype:
                cset (list): A list of smiles that contains the vocabulary for the training data. 

        """
        cset = set()
        for smiles in list_smiles:
            mol = fast_jtnn.MolTree(smiles)
            for c in mol.nodes:
                cset.add(c.smiles)
       # cset_newline = list(map(lambda x: x + "\n", cset))
        return list(cset)

    def _tensorize(self, smiles, assm=True):
        mol_tree = fast_jtnn.MolTree(smiles)
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
        # TODO In the future, this can be parallelized
        preprocessed = list(
            map(self._tensorize, tqdm(list_smiles, leave=True)))
        return preprocessed

    def train_rand_gen(self, loader, load_epoch, lr, anneal_rate, clip_norm, num_epochs, beta, max_beta, step_beta, anneal_iter, kl_anneal_iter, print_iter, save_iter):
        r"""
            Train the Junction Tree Variational Autoencoder for the random generation task.

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
        vocab = fast_jtnn.Vocab(self.vocab)

        for param in self.vae.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if 0 > 0:
            self.vae.load_state_dict(torch.load(
                'checkpoints' + "/model.iter-" + str(load_epoch)))

        print("Model #Params: %dK" %
              (sum([x.nelement() for x in self.vae.parameters()]) / 1000,))

        optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
        scheduler.step()

        def param_norm(m): return math.sqrt(
            sum([p.norm().item() ** 2 for p in m.parameters()]))
        def grad_norm(m): return math.sqrt(
            sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

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

                meters = meters + \
                    np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

                if total_step % print_iter == 0:
                    meters /= 50
                    print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                        total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(self.vae), grad_norm(self.vae)))
                    sys.stdout.flush()
                    meters *= 0

                if total_step % save_iter == 0:
                    torch.save(self.vae.state_dict(), "saved" +
                               "/model.iter-" + str(total_step))

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

    def train_cons_optim(self, loader, batch_size, num_epochs, hidden_size, latent_size, depth, beta, lr):
        r"""
            Train the Junction Tree Variational Autoencoder for the constrained optimization task.

            Args:
                loader (MolTreeFolder): The MolTreeFolder loader.
                batch_size (int): The batch size.
                num_epochs (int): The number of epochs.
                hidden_size (int): The hidden size.
                latent_size (int): The latent size.
                depth (int): The depth of the network.
                lr (float): The learning rate for training.
                beta (float): The KL regularization weight.

        """
        for param in self.prop_vae.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)

        print("Model #Params: %dK" %
              (sum([x.nelement() for x in self.prop_vae.parameters()]) / 1000,))

        optimizer = optim.Adam(self.prop_vae.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler.step()

        PRINT_ITER = 20

        for epoch in range(num_epochs):
            word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0

            for it, batch in enumerate(tqdm(loader)):
                for mol_tree, _ in batch:
                    for node in mol_tree.nodes:
                        if node.label not in node.cands:
                            node.cands.append(node.label)
                            node.cand_mols.append(node.label_mol)

                self.prop_vae.zero_grad()
                loss, kl_div, wacc, tacc, sacc, dacc, pacc = self.prop_vae(
                    batch, beta)
                loss.backward()
                optimizer.step()

                word_acc += wacc
                topo_acc += tacc
                assm_acc += sacc
                steo_acc += dacc
                prop_acc += pacc

                if (it + 1) % PRINT_ITER == 0:
                    word_acc = word_acc / PRINT_ITER * 100
                    topo_acc = topo_acc / PRINT_ITER * 100
                    assm_acc = assm_acc / PRINT_ITER * 100
                    steo_acc = steo_acc / PRINT_ITER * 100
                    prop_acc /= PRINT_ITER

                    print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Prop: %.4f" % (
                        kl_div, word_acc, topo_acc, assm_acc, steo_acc, prop_acc))
                    word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
                    sys.stdout.flush()

                if (it + 1) % 1500 == 0:  # Fast annealing
                    scheduler.step()
                    print("learning rate: %.6f" % scheduler.get_lr()[0])
#                     torch.save(self.prop_vae.state_dict(), opts.save_path + "/model.iter-%d-%d" % (epoch, it + 1))

            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])
#             torch.save(self.prop_vae.state_dict(), opts.save_path + "/model.iter-" + str(epoch))

    def run_cons_optim(self, list_smiles, sim_cutoff=0.0):
        r"""
        Optimize a set of molecules.

        Args:
            list_smiles (list): List of smiles in training data.
            sim_cutoff (float): Simulation cutoff.

        """
#         data = []
#         with open(opts.test_path) as f:
#             for line in f:
#                 s = line.strip("\r\n ").split()[0]
#                 data.append(s)

        res = []
        for smiles in tqdm(list_smiles):
            mol = Chem.MolFromSmiles(smiles)
            score = Descriptors.MolLogP(
                mol) - jtnn.sascorer.calculateScore(mol)
            new_smiles, sim = self.prop_vae.optimize(
                smiles, sim_cutoff=sim_cutoff, lr=2, num_iter=80)
            new_mol = Chem.MolFromSmiles(new_smiles)
            new_score = Descriptors.MolLogP(
                new_mol) - jtnn.sascorer.calculateScore(new_mol)
            res.append((new_score - score, sim, score,
                       new_score, smiles, new_smiles))
            print(new_score - score, sim, score, new_score, smiles, new_smiles)
        print(sum([x[0] for x in res]), sum([x[1] for x in res]))
