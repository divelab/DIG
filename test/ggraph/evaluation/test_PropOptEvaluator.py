from dig.ggraph.evaluation import PropOptEvaluator
from rdkit import Chem
import shutil

def test_PropOptEvaluator():
    smiles = ['C', 'N', 'O']
    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        mols.append(mol)
    res_dict = {'mols':mols}
    evaluator = PropOptEvaluator()
    results = evaluator.eval(res_dict)
    
    assert results == {1: ('O', -5.496546478798415), 2: ('N', -5.767617318560561), 3: ('C', -6.229620227953575)}


if __name__ == '__main__':
    test_PropOptEvaluator()