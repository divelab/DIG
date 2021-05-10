from dig.ggraph.evaluation import Prop_Optim_Evaluator
from rdkit import Chem
import shutil

def test_Prop_Optim_Evaluator():
    smiles = ['C', 'N', 'O']
    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        mols.append(mol)
    res_dict = {'mols':mols}
    evaluator = Prop_Optim_Evaluator()
    results = evaluator.eval(res_dict)
    
    assert results == {1: ('O', -5.496546478798415), 2: ('N', -5.767617318560561), 3: ('C', -6.229620227953575)}


if __name__ == '__main__':
    test_Prop_Optim_Evaluator()