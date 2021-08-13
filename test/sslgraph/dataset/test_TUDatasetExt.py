import shutil
from dig.sslgraph.dataset import TUDatasetExt

def test_TUDatasetExt():
    ## semisupervised
    # NCI1
    root = './dataset/TUDataset'
    dataset = TUDatasetExt(root, name='NCI1', task='semisupervised')
    assert len(dataset) == 4110
    assert dataset.num_features == 37

    assert dataset[0].x.size() == (21, 37)
    assert dataset[0].y.size() == (1,)
    assert dataset[0].edge_index.size() == (2, 42)

    shutil.rmtree(root)

    # # REDDIT-BINARY
    # dataset = TUDatasetExt(root, name='REDDIT-BINARY', task='semisupervised')
    # assert len(dataset) == 2000
    # assert dataset.num_features == 0

    # assert dataset[0].y.size() == (1,)
    # assert dataset[0].edge_index.size() == (2, 480)

    # shutil.rmtree(root)

    ## unsupervised
    # NCI1
    root = './dataset/TUDataset'
    dataset = TUDatasetExt(root, name='NCI1', task='unsupervised')
    assert len(dataset) == 4110
    assert dataset.num_features == 37

    assert dataset[0].x.size() == (21, 37)
    assert dataset[0].y.size() == (1,)
    assert dataset[0].edge_index.size() == (2, 62)

    shutil.rmtree(root)

    # # REDDIT-BINARY
    # dataset = TUDatasetExt(root, name='REDDIT-BINARY', task='unsupervised')
    # assert len(dataset) == 2000
    # assert dataset.num_features == 1

    # assert dataset[0].x.size() == (218, 1)
    # assert dataset[0].y.size() == (1,)
    # assert dataset[0].edge_index.size() == (2, 697)

    # shutil.rmtree(root)

if __name__ == '__main__':
    test_TUDatasetExt()