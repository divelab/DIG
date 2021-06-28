from rdkit import Chem
import numpy as np
from dig.ggraph.utils import check_chemical_validity, qed, calculate_min_plogp, reward_target_molecule_similarity



class RandGenEvaluator:
    r"""
    Evaluator for random generation task. Metric is validity ratio, uniqueness ratio, and novelty ratio (all represented in percentage).
    """

    def __init__(self):
        pass

    @staticmethod
    def eval(input_dict):
        r"""Run evaluation in random generation task. Compute the validity ratio, uniqueness ratio and novelty ratio of generated molecules (all represented in percentage).

        Args:
            input_dict (dict): A python dict with the following items:
                "mols" --- the list of generated molecules reprsented by rdkit Chem.RWMol or Chem.Mol objects;
                "train_smiles" --- the list of SMILES strings used for training.
            
        :rtype: :class:`dict` a python dict with the following items:
                    "valid_ratio" --- validity percentage;
                    "unique_ratio" --- uniqueness percentage;
                    "novel_ratio" --- novelty percentage.
        """

        mols = input_dict['mols']
        results = {}
        valid_mols = [mol for mol in mols if check_chemical_validity(mol)]
        print("Valid Ratio: {}/{} = {:.2f}%".format(len(valid_mols), len(mols), len(valid_mols)/len(mols)*100))
        results['valid_ratio'] = len(valid_mols) / len(mols) * 100
        
        if len(valid_mols) > 0:
            valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
            unique_smiles = list(set(valid_smiles))
            print("Unique Ratio: {}/{} = {:.2f}%".format(len(unique_smiles), len(valid_smiles), len(unique_smiles)/len(valid_smiles)*100))
            results['unique_ratio'] = len(unique_smiles) / len(valid_smiles) * 100

            if 'train_smiles' in input_dict.keys() and input_dict['train_smiles'] is not None:
                train_smiles = input_dict['train_smiles']
                novels = [1 for smile in valid_smiles if smile not in train_smiles]
                print("Novel Ratio: {}/{} = {:.2f}%".format(len(novels), len(valid_smiles), len(novels)/len(valid_smiles)*100))
                results['novel_ratio'] = len(novels) / len(valid_smiles) * 100
        
        return results


class PropOptEvaluator:
    r"""
    Evaluator for property optimization task. Metric is top-3 property scores among generated molecules.

    Args:
        prop_name (str): A string indicating the name of the molecular property, use 'plogp' for penalized logP or 'qed' for 
            Quantitative Estimate of Druglikeness (QED). (default: :obj:`plogp`)
    """

    def __init__(self, prop_name='plogp'):
        assert prop_name in ['plogp', 'qed']
        self.prop_name = prop_name
    
    def eval(self, input_dict):
        r""" Run evaluation in property optimization task. Find top-3 molucules which have highest property scores.
        
        Args:
            input_dict (dict): A python dict with the following items:
                "mols" --- a list of generated molecules reprsented by rdkit Chem.Mol or Chem.RWMol objects.
            
        :rtype: :class:`dict` a python dict with the following items:
                    1 --- information of molecule with the highest property score;
                    2 --- information of molecule with the second highest property score;
                    3 --- information of molecule with the third highest property score.
                    The molecule information is given in the form of a tuple (SMILES string, property score).
        """

        mols = input_dict['mols']
        prop_fn = qed if self.prop_name == 'qed' else calculate_min_plogp

        results = {}
        valid_mols = [mol for mol in mols if check_chemical_validity(mol)]
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        props = [prop_fn(mol) for mol in valid_mols]
        sorted_index = np.argsort(props)[::-1]
        
        assert len(valid_mols) >= 3
        for i in range(3):
            print("Top {} property score: {}".format(i+1, props[sorted_index[i]]))
            results[i+1] = (valid_smiles[sorted_index[i]], props[sorted_index[i]])
        
        return results


class ConstPropOptEvaluator:
    r"""
    Evaluator for constrained optimization task. Metric is the average property improvements, similarities and success rates under the similarity threshold 0.0, 0.2, 0.4, 0.6.
    """

    def __init__(self):
        pass

    @staticmethod
    def eval(input_dict):
        r""" Run evaluation in constrained optimization task. Compute the average property improvements, similarities and success rates under the similarity threshold 0.0, 0.2, 0.4, 0.6.
        
        Args:
            input_dict (dict): A python dict with the following items:
                "mols_0", "mols_2", "mols_4", "mols_6" --- the list of optimized molecules under the similarity threshold 0.0, 0.2, 0.4, 0.6, all represented by rdkit Chem.RWMol or Chem.Mol objects;
                "inp_smiles" --- the list of SMILES strings of input molecules to be optimized.
            
        :rtype: :class:`dict` a python dict with the following items:
                    0, 2, 4, 6 --- the metric values under the similarity threshold 0.0, 0.2, 0.4, 0.6.
                    The metric values are given in the form of a tuple (success rate, mean of similarity, standard deviation of similarity,
                    mean of property improvement, standard deviation of property improvement).
        """
        
        inp_smiles = input_dict['inp_smiles']
        inp_mols = [Chem.MolFromSmiles(s) for s in inp_smiles]
        inp_props = [calculate_min_plogp(mol) for mol in inp_mols]
        mols_0, mols_2, mols_4, mols_6 = input_dict['mols_0'], input_dict['mols_2'], input_dict['mols_4'], input_dict['mols_6']
        
        results = {}
        
        sims_0 = [reward_target_molecule_similarity(inp_mols[i], mols_0[i]) for i in range(len(mols_0)) if mols_0[i] is not None]
        sims_2 = [reward_target_molecule_similarity(inp_mols[i], mols_2[i]) for i in range(len(mols_2)) if mols_2[i] is not None]
        sims_4 = [reward_target_molecule_similarity(inp_mols[i], mols_4[i]) for i in range(len(mols_4)) if mols_4[i] is not None]
        sims_6 = [reward_target_molecule_similarity(inp_mols[i], mols_6[i]) for i in range(len(mols_6)) if mols_6[i] is not None]

        imps_0 = [calculate_min_plogp(mols_0[i]) - inp_props[i] for i in range(len(mols_0)) if mols_0[i] is not None]
        imps_2 = [calculate_min_plogp(mols_2[i]) - inp_props[i] for i in range(len(mols_2)) if mols_2[i] is not None]
        imps_4 = [calculate_min_plogp(mols_4[i]) - inp_props[i] for i in range(len(mols_4)) if mols_4[i] is not None]
        imps_6 = [calculate_min_plogp(mols_6[i]) - inp_props[i] for i in range(len(mols_6)) if mols_6[i] is not None]

        suc_rate = len(imps_0) / len(inp_mols) * 100
        sim_avg, sim_std = np.mean(sims_0), np.std(sims_0)
        imp_avg, imp_std = np.mean(imps_0), np.std(imps_0)
        print("Under similarity threshold 0.0, the success rate is {:.2f}%, the average similarity is {:.2f}+-{:.2f}, the average improvement is {:.2f}+-{:.2f}"
            .format(suc_rate, sim_avg, sim_std, imp_avg, imp_std))
        results[0] = (suc_rate, sim_avg, sim_std, imp_avg, imp_std)

        suc_rate = len(imps_2) / len(inp_mols) * 100
        sim_avg, sim_std = np.mean(sims_2), np.std(sims_2)
        imp_avg, imp_std = np.mean(imps_2), np.std(imps_2)
        print("Under similarity threshold 0.2, the success rate is {:.2f}%, the average similarity is {:.2f}+-{:.2f}, the average improvement is {:.2f}+-{:.2f}"
            .format(suc_rate, sim_avg, sim_std, imp_avg, imp_std))
        results[2] = (suc_rate, sim_avg, sim_std, imp_avg, imp_std)

        suc_rate = len(imps_4) / len(inp_mols) * 100
        sim_avg, sim_std = np.mean(sims_4), np.std(sims_4)
        imp_avg, imp_std = np.mean(imps_4), np.std(imps_4)
        print("Under similarity threshold 0.4, the success rate is {:.2f}%, the average similarity is {:.2f}+-{:.2f}, the average improvement is {:.2f}+-{:.2f}"
            .format(suc_rate, sim_avg, sim_std, imp_avg, imp_std))
        results[4] = (suc_rate, sim_avg, sim_std, imp_avg, imp_std)

        suc_rate = len(imps_6) / len(inp_mols) * 100
        sim_avg, sim_std = np.mean(sims_6), np.std(sims_6)
        imp_avg, imp_std = np.mean(imps_6), np.std(imps_6)
        print("Under similarity threshold 0.6, the success rate is {:.2f}%, the average similarity is {:.2f}+-{:.2f}, the average improvement is {:.2f}+-{:.2f}"
            .format(suc_rate, sim_avg, sim_std, imp_avg, imp_std))
        results[6] = (suc_rate, sim_avg, sim_std, imp_avg, imp_std)

        return results
