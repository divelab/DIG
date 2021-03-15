import torch
import torch.nn as nn
from multiprocessing import Pool

import os
import math, random, sys
from optparse import OptionParser
import pickle

#from fast_jtnn import *
import rdkit
from rdkit import RDLogger
from mol_tree import MolTree

def tensorize(smiles, assm=True):
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

if __name__ == "__main__":
    lg = RDLogger.logger() 
    lg.setLevel(RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-d", "--data_file", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    with open(os.path.join(os.pardir, "datasets", opts.train_path), "r") as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    all_data = pool.map(tensorize, data)

    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('moses-processed/tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
