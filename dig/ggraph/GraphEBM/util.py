### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow

import numpy as np
from rdkit import Chem
import re
import torch
import copy
import os

from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem.Descriptors import qed

atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
ATOM_VALENCY = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}


def adj_to_smiles(adj, x, atomic_num_list):
    valid = [Chem.MolToSmiles(construct_mol(x_elem, adj_elem, atomic_num_list), isomericSmiles=True) for x_elem, adj_elem in zip(x, adj)]
    return valid


def construct_mol(x, A, atomic_num_list):
    """
    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    atoms = np.argmax(x, axis=1)
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
#     print('num atoms: {}'.format(len(atoms)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+] [S+]
            # not support [O-], [N-] [S-]  [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence
    
    
def valid_mol(x):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=True)) if x is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def correct_mol(x):
    xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])
    return mol


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol



def gen_mol(adj, x, atomic_num_list, correct_validity=True, largest_connected_comp=True):
    """
    :param adj:  (10000, 4, 9, 9)
    :param x: (10000, 9, 5)
    :param atomic_num_list: [6, 7, 8, 9, 0]
    """
    adj = adj.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    if not correct_validity:
        gen_mols = [valid_mol_can_with_seg(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
    else:
        gen_mols = []
        for x_elem, adj_elem in zip(x, adj):
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            cmol = correct_mol(mol)
            vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
            gen_mols.append(vcmol)

    return gen_mols



def rescale_adj(adj, type='all'):
    
    if type=='view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj  # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
    
    elif type=='all':
    # Adj: (128, 4, 9, 9):
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv.unsqueeze(1).unsqueeze(2) * adj
    return adj_prime

def turn_valid(adj, x, atomic_num_list, return_unique=False, correct_validity=True, largest_connected_comp=True, debug=True):
    """
    :param adj:  (10000, 4, 9, 9)
    :param x: (10000, 9, 5)
    :param atomic_num_list: [6, 7, 8, 9, 0]
    """
    adj = adj.detach().clone().cpu().numpy()
    x = x.detach().clone().cpu().numpy()
    if not correct_validity:
#         valid = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
        valid = [valid_mol_can_with_seg(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
    else:
        valid = []
        for x_elem, adj_elem in zip(x, adj):
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            cmol = correct_mol(mol)
            vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
            valid.append(vcmol)
    
    
    valid_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None for mol in valid]
    valid_mols = [Chem.MolFromSmiles(s) if s is not None else None for s in valid_smiles]
    
    results = dict()
    results['valid_mols'] = valid_mols
    results['valid_smiles'] = valid_smiles

    return results

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

def update_save_dict(save_dict, org_smile, cur_smile, imp, sim, lenth=10):
    if imp <= 0. or sim == 1.0:
        return save_dict
    
    else:
        if org_smile not in save_dict:
            save_dict[org_smile] = [[[], [], []], [[], [], []], [[], [], []], [[], [], []]]
        if sim >= 0.:
            if len(save_dict[org_smile][0][0]) < lenth:
                save_dict[org_smile][0][0].append(cur_smile)
                save_dict[org_smile][0][1].append(imp)
                save_dict[org_smile][0][2].append(sim)
            elif imp > np.min(save_dict[org_smile][0][1]):
                ind = np.argmin(save_dict[org_smile][0][1])
                save_dict[org_smile][0][0][ind] = cur_smile
                save_dict[org_smile][0][1][ind] = imp
                save_dict[org_smile][0][2][ind] = sim

        if sim >= 0.2:
            if len(save_dict[org_smile][1][0]) < lenth:
                save_dict[org_smile][1][0].append(cur_smile)
                save_dict[org_smile][1][1].append(imp)
                save_dict[org_smile][1][2].append(sim)
            elif imp > np.min(save_dict[org_smile][1][1]):
                ind = np.argmin(save_dict[org_smile][1][1])
                save_dict[org_smile][1][0][ind] = cur_smile
                save_dict[org_smile][1][1][ind] = imp
                save_dict[org_smile][1][2][ind] = sim

        if sim >= 0.4:
            if len(save_dict[org_smile][2][0]) < lenth:
                save_dict[org_smile][2][0].append(cur_smile)
                save_dict[org_smile][2][1].append(imp)
                save_dict[org_smile][2][2].append(sim)
            elif imp > np.min(save_dict[org_smile][2][1]):
                ind = np.argmin(save_dict[org_smile][2][1])
                save_dict[org_smile][2][0][ind] = cur_smile
                save_dict[org_smile][2][1][ind] = imp
                save_dict[org_smile][2][2][ind] = sim

        if sim >= 0.6:
            if len(save_dict[org_smile][3][0]) < lenth:
                save_dict[org_smile][3][0].append(cur_smile)
                save_dict[org_smile][3][1].append(imp)
                save_dict[org_smile][3][2].append(sim)
            elif imp > np.min(save_dict[org_smile][3][1]):
                ind = np.argmin(save_dict[org_smile][3][1])
                save_dict[org_smile][3][0][ind] = cur_smile
                save_dict[org_smile][3][1][ind] = imp
                save_dict[org_smile][3][2][ind] = sim 
        return save_dict 

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

def con_draw_mols(total_optim_dict, dir_name, sim=3):
    all_keys = list(total_optim_dict.keys())
    cnt = 0
    for key in all_keys:
        if len(total_optim_dict[key][sim][0]) > 0:
            gen_dir = './saved_images_{}/{}'.format(dir_name, cnt)
            os.makedirs(gen_dir, exist_ok=True)
            cnt += 1
            t_flag = 0
            
#             for i in range(len(total_optim_dict[key][0][0])):
#                 mol = Chem.MolFromSmiles(total_optim_dict[key][0][0][i])
#                 if qed(mol) < 0.8: continue
#                 filepath = os.path.join(gen_dir, 'sim_{}_imp_{}_qed_{}_{}.png'.format(total_optim_dict[key][0][2][i],total_optim_dict[key][0][1][i],qed(mol),total_optim_dict[key][0][0][i]))
#                 img = Draw.MolToImage(mol)
#                 img.save(filepath)
            for i in range(len(total_optim_dict[key][1][0])):
                mol = Chem.MolFromSmiles(total_optim_dict[key][1][0][i])
                if total_optim_dict[key][1][1][i] < 2: continue
                filepath = os.path.join(gen_dir, 'sim_{}_imp_{}_qed_{}_{}.png'.format(total_optim_dict[key][1][2][i],total_optim_dict[key][1][1][i],qed(mol),total_optim_dict[key][1][0][i]))
                img = Draw.MolToImage(mol)
                img.save(filepath)
                t_flag = 1
            for i in range(len(total_optim_dict[key][2][0])):
                mol = Chem.MolFromSmiles(total_optim_dict[key][2][0][i])
                if total_optim_dict[key][2][1][i] < 2: continue
                filepath = os.path.join(gen_dir, 'sim_{}_imp_{}_qed_{}_{}.png.png'.format(total_optim_dict[key][2][2][i],total_optim_dict[key][2][1][i],qed(mol),total_optim_dict[key][2][0][i]))
                img = Draw.MolToImage(mol)
                img.save(filepath)
                t_flag = 1
            for i in range(len(total_optim_dict[key][3][0])):
                mol = Chem.MolFromSmiles(total_optim_dict[key][3][0][i])
                if total_optim_dict[key][3][1][i] < 2: continue
                filepath = os.path.join(gen_dir, 'sim_{}_imp_{}_qed_{}_{}.png.png'.format(total_optim_dict[key][3][2][i],total_optim_dict[key][3][1][i],qed(mol),total_optim_dict[key][3][0][i]))
                img = Draw.MolToImage(mol)
                img.save(filepath)
                t_flag = 1
            if t_flag == 1:
                filepath = os.path.join(gen_dir, 'original_{}.png'.format(key))
                mol = Chem.MolFromSmiles(key)
                img = Draw.MolToImage(mol)
                img.save(filepath)
    return 


