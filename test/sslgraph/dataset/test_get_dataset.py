import shutil
from dig.sslgraph.dataset import get_dataset


def test_get_dataset():
    ## semisupervised
    # NCI1
    root = './dataset/TUDataset'
    dataset, dataset_pretrain = get_dataset(name='NCI1', task='semisupervised', feat_str='deg+odeg100', root=root)
    assert len(dataset) == 4110
    assert len(dataset_pretrain) == 4110
    assert dataset.num_features == 139
    assert dataset_pretrain.num_features == 139

    assert dataset[0].x.size() == (21, 139)
    assert dataset_pretrain[0].x.size() == (21, 139)
    assert dataset[0].y.size() == (1,)
    assert dataset_pretrain[0].y.size() == (1,)
    assert dataset[0].edge_index.size() == (2, 42)
    assert dataset_pretrain[0].edge_index.size() == (2, 42)
    shutil.rmtree(root)

#     # PROTEINS
#     root = './dataset/TUDataset'
#     dataset, dataset_pretrain = get_dataset(name='PROTEINS', task='semisupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 1113
#     assert len(dataset_pretrain) == 1113
#     assert dataset.num_features == 106
#     assert dataset_pretrain.num_features == 106

#     assert dataset[0].x.size() == (42, 106)
#     assert dataset_pretrain[0].x.size() == (42, 106)
#     assert dataset[0].y.size() == (1,)
#     assert dataset_pretrain[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 162)
#     assert dataset_pretrain[0].edge_index.size() == (2,162)
#     shutil.rmtree(root)

#     # DD
#     root = './dataset/TUDataset'
#     dataset, dataset_pretrain = get_dataset(name='DD', task='semisupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 1178
#     assert len(dataset_pretrain) == 1178
#     assert dataset.num_features == 101
#     assert dataset_pretrain.num_features == 101 

#     assert dataset[0].x.size() == (327, 101)
#     assert dataset_pretrain[0].x.size() == (327, 101)
#     assert dataset[0].y.size() == (1,)
#     assert dataset_pretrain[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 1798)
#     assert dataset_pretrain[0].edge_index.size() == (2, 1798)
#     shutil.rmtree(root)

#     # COLLAB
#     root = './dataset/TUDataset'
#     dataset, dataset_pretrain = get_dataset(name='COLLAB', task='semisupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 5000
#     assert len(dataset_pretrain) == 5000
#     assert dataset.num_features == 103
#     assert dataset_pretrain.num_features == 103

#     assert dataset[0].x.size() == (45, 103)
#     assert dataset_pretrain[0].x.size()  == (45, 103)
#     assert dataset[0].y.size() == (1,)
#     assert dataset_pretrain[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 1980)
#     assert dataset_pretrain[0].edge_index.size() == (2, 1980)
#     shutil.rmtree(root)

    # REDDIT-BINARY
    # root = './dataset/TUDataset'
    # dataset, dataset_pretrain = get_dataset(name='REDDIT-BINARY', task='semisupervised', feat_str='deg+odeg100', root=root)
    # assert len(dataset) == 2000
    # assert len(dataset_pretrain) == 2000
    # assert dataset.num_features == 13
    # assert dataset_pretrain.num_features == 13

    # assert dataset[0].x.size() == (218, 13)
    # assert dataset_pretrain[0].x.size() == (218, 13)
    # assert dataset[0].y.size() == (1,)
    # assert dataset_pretrain[0].y.size() == (1,)
    # assert dataset[0].edge_index.size() == (2, 480)
    # assert dataset_pretrain[0].edge_index.size() == (2, 480)
    # shutil.rmtree(root)

#     # REDDIT-MULTI-5K
#     root = './dataset/TUDataset'
#     dataset, dataset_pretrain = get_dataset(name='REDDIT-MULTI-5K', task='semisupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 4999
#     assert len(dataset_pretrain) == 4999
#     assert dataset.num_features == 13
#     assert dataset_pretrain.num_features == 13

#     assert dataset[0].x.size() == (1593, 13)
#     assert dataset_pretrain[0].x.size() == (1593, 13)
#     assert dataset[0].y.size() == (1,)
#     assert dataset_pretrain[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 3776)
#     assert dataset_pretrain[0].edge_index.size() == (2, 3776)
#     shutil.rmtree(root)

    ## unsupervised
    # NCI1
    root = './dataset/TUDataset'
    dataset = get_dataset(name='NCI1', task='unsupervised', feat_str='deg+odeg100', root=root)
    assert len(dataset) == 4110
    assert dataset.num_features == 43

    assert dataset[0].x.size() == (21, 43)
    assert dataset[0].y.size() == (1,)
    assert dataset[0].edge_index.size() == (2, 62)
    shutil.rmtree(root)

#     # PROTEINS
#     root = './dataset/TUDataset'
#     dataset = get_dataset(name='PROTEINS', task='unsupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 1113
#     assert dataset.num_features == 31

#     assert dataset[0].x.size() == (42, 31)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 203)
#     shutil.rmtree(root)

#     # MUTAG
#     root = './dataset/TUDataset'
#     dataset = get_dataset(name='MUTAG', task='unsupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 188
#     assert dataset.num_features == 13

#     assert dataset[0].x.size() == (17, 13)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 54)
#     shutil.rmtree(root)

#     # DD
#     root = './dataset/TUDataset'
#     dataset = get_dataset(name='DD', task='unsupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 1178
#     assert dataset.num_features == 110

#     assert dataset[0].x.size() == (327, 110)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 2124)
#     shutil.rmtree(root)

#     # COLLAB
#     root = './dataset/TUDataset'
#     dataset = get_dataset(name='COLLAB', task='unsupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 5000
#     assert dataset.num_features == 494

#     assert dataset[0].x.size() == (45, 494)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 2024)
#     shutil.rmtree(root)

    # REDDIT-BINARY
    # root = './dataset/TUDataset'
    # dataset = get_dataset(name='REDDIT-BINARY', task='unsupervised', feat_str='deg+odeg100', root=root)
    # assert len(dataset) == 2000
    # assert dataset.num_features == 3065

    # assert dataset[0].x.size() == (218, 3065)
    # assert dataset[0].y.size() == (1,)
    # assert dataset[0].edge_index.size() == (2, 697)
    # shutil.rmtree(root)

#     # REDDIT-MULTI-5K
#     root = './dataset/TUDataset'
#     dataset = get_dataset(name='REDDIT-MULTI-5K', task='unsupervised', feat_str='deg+odeg100', root=root)
#     assert len(dataset) == 4999
#     assert dataset.num_features == 2014

#     assert dataset[0].x.size() == (1593, 2014)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 5368)
#     shutil.rmtree(root)

if __name__ == '__main__':
    test_get_dataset()