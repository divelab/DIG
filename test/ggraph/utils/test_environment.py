from rdkit import Chem
from dig.ggraph.utils import steric_strain_filter, zinc_molecule_filter, convert_radical_electrons_to_hydrogens, check_valency

def test_environment():
    mol = Chem.MolFromSmiles('C')
    mol = convert_radical_electrons_to_hydrogens(mol)

    assert steric_strain_filter(mol)
    assert zinc_molecule_filter(mol)
    assert check_valency(mol)

if __name__ == '__main__':
    test_environment()