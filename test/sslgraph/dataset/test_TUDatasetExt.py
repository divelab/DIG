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

#     # PROTEINS
#     dataset = TUDatasetExt(root, name='PROTEINS', task='semisupervised')
#     assert len(dataset) == 1113
#     assert dataset.num_features == 3

#     assert dataset[0].x.size() == (42, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 162)

#     # DD
#     dataset = TUDatasetExt(root, name='DD', task='semisupervised')
#     assert len(dataset) == 1178
#     assert dataset.num_features == 89

#     assert dataset[0].x.size() == (327, 89)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 1798) 

#     # COLLAB
#     dataset = TUDatasetExt(root, name='COLLAB', task='semisupervised')
#     assert len(dataset) == 5000
#     assert dataset.num_features == 0

#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 1980)

    # REDDIT-BINARY
    dataset = TUDatasetExt(root, name='REDDIT-BINARY', task='semisupervised')
    assert len(dataset) == 2000
    assert dataset.num_features == 0

    assert dataset[0].y.size() == (1,)
    assert dataset[0].edge_index.size() == (2, 480)

#     # REDDIT-MULTI-5K
#     dataset = TUDatasetExt(root, name='REDDIT-MULTI-5K', task='semisupervised')
#     assert len(dataset) == 4999
#     assert dataset.num_features == 0

#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 3776)
#     shutil.rmtree(root)

    ## unsupervised
    # NCI1
    root = './dataset/TUDataset'
    dataset = TUDatasetExt(root, name='NCI1', task='unsupervised')
    assert len(dataset) == 4110
    assert dataset.num_features == 37

#     assert dataset[0].x.size() == (21, 37)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 62)

#     # PROTEINS
#     dataset = TUDatasetExt(root, name='PROTEINS', task='unsupervised')
#     assert len(dataset) == 1113
#     assert dataset.num_features == 3

#     assert dataset[0].x.size() == (42, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 203)

#     # MUTAG
#     dataset = TUDatasetExt(root, name='MUTAG', task='unsupervised')
#     assert len(dataset) == 188
#     assert dataset.num_features == 7

#     assert dataset[0].x.size() == (17, 7)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 54)

#     # DD
#     dataset = TUDatasetExt(root, name='DD', task='unsupervised')
#     assert len(dataset) == 1178
#     assert dataset.num_features == 89

#     assert dataset[0].x.size() == (327, 89)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 2124)

#     # COLLAB
#     dataset = TUDatasetExt(root, name='COLLAB', task='unsupervised')
#     assert len(dataset) == 5000
#     assert dataset.num_features == 1

#     assert dataset[0].x.size() == (45, 1)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 2024)

    # REDDIT-BINARY
    dataset = TUDatasetExt(root, name='REDDIT-BINARY', task='unsupervised')
    assert len(dataset) == 2000
    assert dataset.num_features == 1

    assert dataset[0].x.size() == (218, 1)
    assert dataset[0].y.size() == (1,)
    assert dataset[0].edge_index.size() == (2, 697)

#     # REDDIT-MULTI-5K
#     dataset = TUDatasetExt(root, name='REDDIT-MULTI-5K', task='unsupervised')
#     assert len(dataset) == 4999
#     assert dataset.num_features == 1

#     assert dataset[0].x.size() == (1593, 1)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].edge_index.size() == (2, 5368)
#     shutil.rmtree(root)

if __name__ == '__main__':
    test_TUDatasetExt()