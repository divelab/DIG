from dig.ggraph.dataset import MOSES

def test_moses():
    dataset = MOSES(root='./dataset/MOSES')

    assert len(dataset) == 1936962
    assert dataset.num_features == 9
    assert dataset.__repr__() == 'moses(1936962)'

    assert len(dataset[0]) == 5
    assert dataset[0].x.size() == (38, 9)
    assert dataset[0].adj.size() == (4, 38, 38)
    assert dataset[0].bfs_perm_origin.size() == (38,)
    assert dataset[0].num_atom.size() == (1,)