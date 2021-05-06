from dig.ggraph.evaluation import Cons_Optim_Evaluator
from rdkit import Chem
import shutil

def test_Cons_Optim_Evaluator():
    smile = 'C'
    mol = Chem.MolFromSmiles(smile)
    res_dict = {'inp_smiles': smile, 'mols_0':[mol], 'mols_2': [mol], 'mols_4': [mol], 'mols_6': [mol]}

    evaluator = Cons_Optim_Evaluator()
    results = evaluator.eval(res_dict)
    
    assert results == {0: (100.0, 1.0, 0.0, 0.0, 0.0), 2: (100.0, 1.0, 0.0, 0.0, 0.0), 4: (100.0, 1.0, 0.0, 0.0, 0.0), 6: (100.0, 1.0, 0.0, 0.0, 0.0)}

if __name__ == '__main__':
    test_Cons_Optim_Evaluator()