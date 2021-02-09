from .environment import check_chemical_validity
from .data_io import get_smiles
from rdkit import Chem



def metric_random_generation(mols, data_file=None):
    results = {}
    valid_mols = [mol for mol in mols if check_chemical_validity(mol)]
    print("Valid Ratio: {}/{} = {:.2f}%".format(len(valid_mols), len(mols), len(valid_mols)/len(mols)*100))
    results['valid_ratio'] = len(valid_mols) / len(mols) * 100
    
    if len(valid_mols) > 0:
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        unique_smiles = list(set(valid_smiles))
        print("Unique Ratio: {}/{} = {:.2f}%".format(len(unique_smiles), len(valid_smiles), len(unique_smiles)/len(valid_smiles)*100))
        results['unique_ratio'] = len(unique_smiles) / len(valid_smiles) * 100

        if data_file is not None:
            train_smiles = get_smiles(data_file)
            novels = [1 for smile in valid_smiles if smile not in train_smiles]
            print("Novel Ratio: {}/{} = {:.2f}%".format(len(novels), len(train_smiles), len(novels)/len(train_smiles)*100))
            results['novel_ratio'] = len(novels) / len(valid_smiles) * 100
    
    return results