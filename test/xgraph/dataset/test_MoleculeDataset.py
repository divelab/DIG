from dig.xgraph.dataset import MoleculeDataset
import shutil

def test_MoleculeDataset():
    root = 'datasets'
    dataset = MoleculeDataset(root, 'Tox21')



    shutil.rmtree(root)





if __name__ == '__main__':
    test_MoleculeDataset()