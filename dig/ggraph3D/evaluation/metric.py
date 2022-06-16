import torch
import numpy as np
from dig.ggraph3D.utils import xyz2mol, collect_bond_dists, compute_mmd, compute_prop



class RandGenEvaluator:
    
    def __init__(self):
        pass

    @staticmethod
    def eval_validity(mol_dicts):
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
    def __init__(self, prop_name='gap', good_threshold=4.5):
        assert prop_name in ['gap', 'alpha']
        self.prop_name = prop_name
        self.good_threshold = good_threshold
    
    def eval(self, mol_dicts):
        results = {}

        prop_list = []
        for num_atoms in mol_dicts:
            atomic_numbers, positions = mol_dicts[num_atoms]['_atomic_numbers'], mol_dicts[num_atoms]['_positions']
            for atomic_number, position in zip(atomic_numbers, positions):
                _, valid = xyz2mol(atomic_number, position)
                if not valid:
                    continue
                prop_list.append(compute_prop(atomic_number, position, self.prop_name))
        
        mean, median = np.mean(prop_list), np.median(prop_list)
        if self.prop_name == 'gap':
            best = np.min(prop_list)
            good_per = np.sum(np.array(prop_list) <= self.good_threshold) / len(prop_list)
        elif self.prop_name == 'alpha':
            best = np.max(prop_list)
            good_per = np.sum(np.array(prop_list) >= self.good_threshold) / len(prop_list)
        
        print("Mean: {}, median: {}, best: {}, good percentage: {:.2f}".format(mean, median, best, good_per * 100))
        
        results['mean'] = mean
        results['median'] = median
        results['best'] = best
        results['good_per'] = good_per
        return results