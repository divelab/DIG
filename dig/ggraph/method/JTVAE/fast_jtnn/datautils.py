import pickle
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .mol_tree import MolTree
from .jtnn_enc import JTNNEncoder
from .mpn import MPN
from .jtmpn import JTMPN
from .vocab import Vocab


class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [data[i: i + self.batch_size]
                       for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                    num_workers=self.num_workers, collate_fn=lambda x: x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class MolTreeFolder(object):

    def __init__(self, preprocessed_data, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.preprocessed_data = preprocessed_data
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        #         for fn in self.data_files:

        if self.shuffle:
            random.shuffle(self.preprocessed_data)  # shuffle data before batch

        batches = [self.preprocessed_data[i: i + self.batch_size]
                   for i in range(0, len(self.preprocessed_data), self.batch_size)]
        if len(batches[-1]) < self.batch_size:
            batches.pop()

        dataset = MolTreeDataset(batches, self.vocab, self.assm)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=self.num_workers, collate_fn=lambda x: x[0])

        for b in dataloader:
            yield b

        del self.preprocessed_data, batches, dataset, dataloader


class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)


class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = Vocab(vocab)
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)


def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            # Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree


class PropDataset(Dataset):

    def __init__(self, data, prop_values):
        self.prop_data = prop_values
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
