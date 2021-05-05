from dig.ggraph.dataset.ggraph_dataset import ZINC250k

def test_zinc250k():
    dataset = ZINC250k(root='./dataset', prop_name='penalized_logp')
    
    assert len(dataset) == 249455
    assert dataset.num_features == 9
    assert dataset.__repr__() == 'zinc250k_property(249455)'

    assert dataset[0].x.size() == (38, 9)
    assert dataset[0].adj.size() == (4, 38, 38)
    assert dataset[0].bfs_perm_origin.size() == (38,)
    assert dataset[0].num_atom.size() == (1,)

