from dig.oodgraph import GOODCBAS, GOODCMNIST, GOODCora, GOODHIV, GOODMotif, GOODPCBA, GOODZINC, GOODArxiv
import pytest
import shutil

dataset_domain = {
    'GOODHIV': ['scaffold', 'size'],
    'GOODPCBA': ['scaffold', 'size'],
    'GOODZINC': ['scaffold', 'size'],
    'GOODCMNIST': ['color'],
    'GOODMotif': ['basis', 'size'],
    'GOODCora': ['word', 'degree'],
    'GOODArxiv': ['time', 'degree'],
    'GOODCBAS': ['color']
}

@pytest.mark.parametrize('dataset_name', list(dataset_domain.keys()))
def test_dataset(dataset_name):
    root = 'datasets'
    for shift_type in ['no_shift', 'covariate', 'concept']:
        for domain in dataset_domain[dataset_name]:
            dataset, meta_info = eval(dataset_name).load(root, domain, shift=shift_type)
            assert dataset is not None
            assert meta_info is not None
    shutil.rmtree(root)