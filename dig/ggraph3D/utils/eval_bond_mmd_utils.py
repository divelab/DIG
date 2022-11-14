import torch
import numpy as np


def collect_bond_dists(mols_dict, valid_list, con_mat_list):
    """
    Collect the lengths for each type of chemical bond in given valid molecular geometries.

    Args:
        mol_dicts (dict): A python dict where the key is the number of atoms, and the value indexed by that key is another python dict storing the atomic
            number matrix (indexed by the key '_atomic_numbers') and the coordinate tensor (indexed by the key '_positions') of all generated molecular geometries with that atom number.
        valid_list (list): the list of bool values indicating whether each molecular geometry is chemically valid. Note that only the bond lengths of 
            valid molecular geometries will be collected.
        con_mat_list (list): the list of bond order matrices.
    
    :rtype: :class:`dict` a python dict where the key is the bond type, and the value indexed by that key is the list of all bond lengths of that bond.
    """
    bond_dist = {}
    id = 0

    for n_atoms in mols_dict:
        numbers, positions = mols_dict[n_atoms]['_atomic_numbers'], mols_dict[n_atoms]['_positions']
        for pos, num in zip(positions, numbers):
            if not valid_list[id]:
                id += 1
                continue
            
            atom_ids_1, atom_ids_2 = np.nonzero(con_mat_list[id])
            for id1, id2 in zip(atom_ids_1, atom_ids_2):
                if id1 < id2:
                    continue
                atomic1, atomic2 = num[id1], num[id2]
                z1, z2 = min(atomic1, atomic2), max(atomic1, atomic2)
                bond_type = con_mat_list[id][id1, id2]

                if not (z1, z2, bond_type) in bond_dist:
                    bond_dist[(z1, z2, bond_type)] = []
                bond_dist[(z1, z2, bond_type)].append(np.linalg.norm(pos[id1]-pos[id2]))
            id += 1
    
    return bond_dist


def compute_mmd(source, target, batch_size=1000, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the `maximum mean discrepancy distance <https://jmlr.csail.mit.edu/papers/v13/gretton12a.html>`_ between two sample set.
    This implementation is based on `this open source code <https://github.com/ZongxianLee/MMD_Loss.Pytorch>`_.

    Args:
        source (pytorch tensor): the pytorch tensor containing data samples of the source distribution.
        target (pytorch tensor): the pytorch tensor containing data samples of the target distribution.

    :rtype:
        :class:`float`
    """
    n_source = int(source.size()[0])
    n_target = int(target.size()[0])
    n_samples = n_source + n_target
    
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth, id = 0.0, 0
        while id < n_samples:
            bandwidth += torch.sum((total0-total1[id:id+batch_size])**2)
            id += batch_size
        bandwidth /= n_samples ** 2 - n_samples
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    XX_kernel_val = [0 for _ in range(kernel_num)]
    for i in range(kernel_num):
        XX_kernel_val[i] += torch.sum(torch.exp(-((total0[:,:n_source] - total1[:n_source,:])**2) / bandwidth_list[i]))
    XX = sum(XX_kernel_val) / (n_source * n_source)

    YY_kernel_val = [0 for _ in range(kernel_num)]
    id = n_source
    while id < n_samples:
        for i in range(kernel_num):
            YY_kernel_val[i] += torch.sum(torch.exp(-((total0[:,n_source:] - total1[id:id+batch_size,:])**2) / bandwidth_list[i]))
        id += batch_size
    YY = sum(YY_kernel_val) / (n_target * n_target)

    XY_kernel_val = [0 for _ in range(kernel_num)]
    id = n_source
    while id < n_samples:
        for i in range(kernel_num):
            XY_kernel_val[i] += torch.sum(torch.exp(-((total0[:,id:id+batch_size] - total1[:n_source,:])**2) / bandwidth_list[i]))
        id += batch_size
    XY = sum(XY_kernel_val) / (n_source * n_target)
    
    return XX.item() + YY.item() - 2 * XY.item()