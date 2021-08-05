from dig.ggraph.utils import gen_mol_from_one_shot_tensor
import torch
from rdkit import Chem

def test_gen_mol_from_one_shot_tensor():
    adj = torch.ones(1, 4, 6, 6)
    x = torch.ones(1, 4, 6)
    atomic_num_list = [6, 7, 8, 0]
    gen_mols = gen_mol_from_one_shot_tensor(adj, x, atomic_num_list)
    gen_smiles = Chem.MolToSmiles(gen_mols[0], isomericSmiles=True)
    
    assert gen_smiles=='C123C45C16C24C356'
    
if __name__ == '__main__':
    test_gen_mol_from_one_shot_tensor()