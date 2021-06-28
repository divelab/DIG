from dig.ggraph.evaluation import RandGenEvaluator
from rdkit import Chem
import shutil

def test_RandGenEvaluator():
    smile = 'CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1'
    mol = Chem.MolFromSmiles(smile)
    res_dict = {'mols':[mol], 'train_smiles': [smile]}
    evaluator = RandGenEvaluator()
    results = evaluator.eval(res_dict)
    
    assert results == {'valid_ratio': 100.0, 'unique_ratio': 100.0, 'novel_ratio': 0.0}


if __name__ == '__main__':
    test_RandGenEvaluator()