import torch
import numpy as np
from dig.ggraph3D.utils import xyz2mol, collect_bond_dists, compute_mmd, compute_prop



class RandGenEvaluator:
    r"""
    Evaluator for random generation task. Metric is the chemical validity ratio (represented in percentage) and the MMD distances of bond length distribution between the generated molecular geometries and those in the dataset.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def eval_validity(mol_dicts):
        r"""Run evaluation in random generation task. Compute the chemical validity ratio (represented in percentage).

        Args:
            mol_dicts (dict): A python dict where the key is the number of atoms, and the value indexed by that key is another python dict storing the atomic
                number matrix (indexed by the key '_atomic_numbers') and the coordinate tensor (indexed by the key '_positions') of all generated molecular geometries with that atom number. 
            
        :rtype: :class:`dict` a python dict with the following items:
                    "valid_ratio" --- chemical validity percentage;
        """

        num_generated, num_valid = 0, 0
        results = {}

        for num_atoms in mol_dicts:
            atomic_numbers, positions = mol_dicts[num_atoms]['_atomic_numbers'], mol_dicts[num_atoms]['_positions']
            num_generated += len(atomic_numbers)

            for atomic_number, position in zip(atomic_numbers, positions):
                _, valid = xyz2mol(atomic_number, position)
                num_valid += 1 if valid else 0
        
        print("Valid Ratio: {}/{} = {:.2f}%".format(num_valid, num_generated, num_valid/num_generated*100))
        results['valid_ratio'] = num_valid / num_generated * 100

        return results
    
    @staticmethod
    def eval_bond_mmd(input_dict):
        r"""Run evaluation in random generation task. Compute the MMD distances of bond length distribution between the generated molecular geometries and those in the dataset.

        Args:
            input_dict (dict): A python dict with the following items:
                "mol_dicts" --- A python dict where the key is the number of atoms, and the value indexed by that key is another python dict storing the atomic
                number matrix (indexed by the key '_atomic_numbers') and the coordinate tensor (indexed by the key '_positions') of all generated molecular geometries with that atom number.
                "target_bond_dists" --- A python dict where the key is the bond type, and the value indexed by that key is the list of all lengths of that bond in the dataset.
            
        :rtype: :class:`dict` a python dict where the key is the bond type, and the value indexed by that key is the MMD distance metric of that bond.
        """

        mol_dicts, target_bond_dists = input_dict['mol_dicts'], input_dict['target_bond_dists']
        bond_types = [(1,8,1),(1,7,1),(6,7,1),(6,8,1),(6,6,1),(1,6,1)]
        atom_type_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
        results = {}

        valid_list, con_mat_list = [], []
        for num_atoms in mol_dicts:
            atomic_numbers, positions = mol_dicts[num_atoms]['_atomic_numbers'], mol_dicts[num_atoms]['_positions']
            for atomic_number, position in zip(atomic_numbers, positions):
                con_mat, valid = xyz2mol(atomic_number, position)
                valid_list.append(valid)
                con_mat_list.append(con_mat)
        
        source_bond_dists = collect_bond_dists(mol_dicts, valid_list, con_mat_list)

        for bond_type in bond_types:
            if bond_type in source_bond_dists:
                mmd = compute_mmd(torch.tensor(source_bond_dists[bond_type]), torch.tensor(target_bond_dists[bond_type]))
                print("The MMD distance of {}-{} bond length distributions is {}".format(
                    atom_type_to_symbol[bond_type[0]], atom_type_to_symbol[bond_type[1]], mmd))
                results[bond_type] = mmd
        
        return results


class PropOptEvaluator:
    r"""
    Evaluator for targeted molecule discovery task. Metric is the mean, optimum, and good percentage of property scores among generated molecular geometries.

    Args:
        prop_name (str): A string indicating the name of the molecular property, use 'gap' for HOMO-LUMO gap or 'alpha' for 
            isotropic polarizability. (default: :obj:`gap`)
        good_threshold (float): The threshold used to identifying whether a property score is good or not. Use 4.5 for HOMO-LUMO gap, and lower than 4.5 is considered
            as good. Use 91 for isotropic polarizability, and higher than 91 is considered as good.
    """

    def __init__(self, prop_name='gap', good_threshold=4.5):
        assert prop_name in ['gap', 'alpha']
        self.prop_name = prop_name
        self.good_threshold = good_threshold
    
    def eval(self, mol_dicts):
        r""" Run evaluation in the targeted molecule discovery task.
        
        Args:
            mol_dicts (dict): A python dict where the key is the number of atoms, and the value indexed by that key is another python dict storing the atomic
                number matrix (indexed by the key '_atomic_numbers') and the coordinate tensor (indexed by the key '_positions') of all generated molecular geometries with that atom number. 
            
        :rtype: :class:`dict` a python dict with the following items:
                    mean --- the mean of property scores;
                    best --- the optimum of property scores, it is the lowest value for HOMO-LUMO gap, and the highest value for isotropic polarizability;
                    good_per --- the good percentage of property scores.
        """
        
        results = {}

        prop_list = []
        for num_atoms in mol_dicts:
            atomic_numbers, positions = mol_dicts[num_atoms]['_atomic_numbers'], mol_dicts[num_atoms]['_positions']
            for atomic_number, position in zip(atomic_numbers, positions):
                _, valid = xyz2mol(atomic_number, position)
                if not valid:
                    continue
                prop_list.append(compute_prop(atomic_number, position, self.prop_name))
        
        mean = np.mean(prop_list)
        if self.prop_name == 'gap':
            best = np.min(prop_list)
            good_per = np.sum(np.array(prop_list) <= self.good_threshold) / len(prop_list)
        elif self.prop_name == 'alpha':
            best = np.max(prop_list)
            good_per = np.sum(np.array(prop_list) >= self.good_threshold) / len(prop_list)
        
        print("Mean: {}, best: {}, good percentage: {:.2f}".format(mean, best, good_per * 100))
        
        results['mean'] = mean
        results['best'] = best
        results['good_per'] = good_per
        return results