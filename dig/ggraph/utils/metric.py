from .environment import check_chemical_validity, qed, calculate_min_plogp, reward_target_molecule_similarity
from .data_io import get_smiles_props_800
from rdkit import Chem
import numpy as np



def metric_random_generation(mols, train_smiles=None):
    """
    Evaluation in random generation task.
    Compute the valid ratio, unique ratio and novel ratio of generated molecules
    param mols: a list of generated molecules reprsented by Chem.RWMol objects
    param train_smiles: a list of smiles strings used for training
    """
    results = {}
    valid_mols = [mol for mol in mols if check_chemical_validity(mol)]
    print("Valid Ratio: {}/{} = {:.2f}%".format(len(valid_mols), len(mols), len(valid_mols)/len(mols)*100))
    results['valid_ratio'] = len(valid_mols) / len(mols) * 100
    
    if len(valid_mols) > 0:
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        unique_smiles = list(set(valid_smiles))
        print("Unique Ratio: {}/{} = {:.2f}%".format(len(unique_smiles), len(valid_smiles), len(unique_smiles)/len(valid_smiles)*100))
        results['unique_ratio'] = len(unique_smiles) / len(valid_smiles) * 100

        if train_smiles is not None:
            novels = [1 for smile in valid_smiles if smile not in train_smiles]
            print("Novel Ratio: {}/{} = {:.2f}%".format(len(novels), len(valid_smiles), len(novels)/len(valid_smiles)*100))
            results['novel_ratio'] = len(novels) / len(valid_smiles) * 100
    
    return results


def metric_property_optimization(mols, topk=3, prop='plogp'):
    """
    Evaluation in property optimization task.
    Find topk molucules which have highest property scores.
    param mols: a list of generated molecules reprsented by Chem.RWMol objects
    """
    assert prop in ['plogp','qed']
    prop_fn = qed if prop == 'qed' else calculate_min_plogp

    results = {}
    valid_mols = [mol for mol in mols if check_chemical_validity(mol)]
    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    props = [prop_fn(mol) for mol in valid_mols]
    sorted_index = np.argsort(props)[::-1]
    
    assert len(valid_mols) >= topk
    for i in range(topk):
        print("Top {} property score: {}".format(i+1, props[sorted_index[i]]))
        results[i] = (valid_smiles[sorted_index[i]], props[sorted_index[i]])
    
    return results


def metric_constrained_optimization(mols_0, mols_2, mols_4, mols_6, data_file=None):
    """
    Evaluation in constrained optimization task.
    Compute the average property improvements, similarities and success rates under the similarity threshold 0.0, 0.2, 0.4, 0.6
    param mols_0, mols_2, mols_4, mols_6: the list of optimized molecules under the similarity threshold 0.0, 0.2, 0.4, 0.6, all represented by Chem.RWMol objects
    """
    assert data_file is not None
    inp_smiles, inp_props = get_smiles_props_800(data_file)
    inp_mols = [Chem.MolFromSmiles(s) for s in inp_smiles]

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
