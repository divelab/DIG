import sys
import os

import numpy as np
import networkx as nx
import random

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

import torch
import torch.nn.functional as F

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

    
def get_maxlen_of_bfs_queue(path):
    """
    Calculate the maxlen of bfs queue.
    """
    fp = open(path, 'r')
    max_all = []
    cnt = 0
    for smiles in fp:
        cnt += 1
        if cnt % 10000 == 0:
            print('cur cnt %d' % cnt)
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        #adj = construct_adj_matrix(mol)
        graph = mol_to_nx(mol)
        N = len(graph.nodes)
        for i in range(N):
            start = i
            order, max_ = bfs_seq(graph, start)
            max_all.append(max_)
    print(max(max_all))


def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        
    print('set seed for random numpy and torch')


def save_one_mol(path, smile, cur_iter=None, score=None):
    """
    save one molecule
    mode: append
    """
    cur_iter = str(cur_iter)

    fp = open(path, 'a')
    fp.write('%s  %s  %s\n' % (cur_iter, smile, str(score)))
    fp.close()


def save_one_reward(path, reward, score, loss, cur_iter):
    """
    save one iter reward/score
    mode: append
    """
    fp = open(path, 'a')
    fp.write('cur_iter: %d | reward: %.5f | score: %.5f | loss: %.5f\n' % (cur_iter, reward, score, loss))
    fp.close()

def save_one_optimized_molecule(path, org_smile, optim_smile, optim_plogp, cur_iter, ranges, sim):
    """
    path: save path
    org_smile: molecule to be optimized
    org_plogp: original plogp
    optim_smile: with shape of (4, ), containing optimized smiles with similarity constrained 0(0.2/0.4/0.6) 
    optim_plogp:  corespongding plogp 
    cur_iter: molecule index

    """
    start = ranges[0]
    end = ranges[1]
    fp1 = open(path + '/sim0_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp2 = open(path + '/sim2_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp3 = open(path + '/sim4_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp4 = open(path + '/sim6_%d_%d' % (ranges[0], ranges[1]), 'a')
    out_string1 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[0], optim_plogp[0], sim[0])
    out_string2 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[1], optim_plogp[1], sim[1])
    out_string3 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[2], optim_plogp[2], sim[2])
    out_string4 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[3], optim_plogp[3], sim[3])

    fp1.write(out_string1)
    fp2.write(out_string2)
    fp3.write(out_string3)
    fp4.write(out_string4)
    #fp.write('cur_iter: %d | reward: %.5f | score: %.5f | loss: %.5f\n' % (cur_iter, reward, score, loss))
    fp1.close()
    fp2.close()
    fp3.close()
    fp4.close()


def update_optim_dict(optim_dict, org_smile, cur_smile, imp, sim):
    if imp <= 0. or sim == 1.0:
        return optim_dict
    
    else:
        if org_smile not in optim_dict:
            optim_dict[org_smile] = [['', -100, -1], ['', -100, -1], ['', -100, -1], ['', -100, -1]]
        if sim >= 0.:
            if imp > optim_dict[org_smile][0][1]:
                optim_dict[org_smile][0][0] = cur_smile
                optim_dict[org_smile][0][1] = imp
                optim_dict[org_smile][0][2] = sim

        if sim >= 0.2:
            if imp > optim_dict[org_smile][1][1]:
                optim_dict[org_smile][1][0] = cur_smile
                optim_dict[org_smile][1][1] = imp
                optim_dict[org_smile][1][2] = sim

        if sim >= 0.4:
            if imp > optim_dict[org_smile][2][1]:
                optim_dict[org_smile][2][0] = cur_smile
                optim_dict[org_smile][2][1] = imp
                optim_dict[org_smile][2][2] = sim

        if sim >= 0.6:
            if imp > optim_dict[org_smile][3][1]:
                optim_dict[org_smile][3][0] = cur_smile
                optim_dict[org_smile][3][1] = imp
                optim_dict[org_smile][3][2] = sim  
        return optim_dict                          


def update_total_optim_dict(total_optim_dict, optim_dict):
    all_keys = list(optim_dict.keys())
    for key in all_keys:
        if key not in total_optim_dict:
            total_optim_dict[key] = [['', -100, -1], ['', -100, -1], ['', -100, -1], ['', -100, -1]]
        
        if optim_dict[key][0][1] > total_optim_dict[key][0][1]:
            total_optim_dict[key][0][0] = optim_dict[key][0][0]
            total_optim_dict[key][0][1] = optim_dict[key][0][1]
            total_optim_dict[key][0][2] = optim_dict[key][0][2]

        if optim_dict[key][1][1] > total_optim_dict[key][1][1]:
            total_optim_dict[key][1][0] = optim_dict[key][1][0]
            total_optim_dict[key][1][1] = optim_dict[key][1][1]
            total_optim_dict[key][1][2] = optim_dict[key][1][2]

        if optim_dict[key][2][1] > total_optim_dict[key][2][1]:
            total_optim_dict[key][2][0] = optim_dict[key][2][0]
            total_optim_dict[key][2][1] = optim_dict[key][2][1]
            total_optim_dict[key][2][2] = optim_dict[key][2][2]

        if optim_dict[key][3][1] > total_optim_dict[key][3][1]:
            total_optim_dict[key][3][0] = optim_dict[key][3][0]
            total_optim_dict[key][3][1] = optim_dict[key][3][1]
            total_optim_dict[key][3][2] = optim_dict[key][3][2]
    return total_optim_dict


def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target,
                                                            radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)