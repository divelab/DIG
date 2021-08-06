from dig.ggraph.dataset import ZINC250k
import shutil

def test_zinc250k():
    root = './dataset/ZINC250k'

    dataset = ZINC250k(root, prop_name='penalized_logp')
    
    assert len(dataset) == 249455
    assert dataset.num_features == 9
    assert dataset.__repr__() == 'zinc250k_property(249455)'
    assert len(dataset.get_split_idx()) == 2

    assert len(dataset[0]) == 6
    assert dataset[0].x.size() == (38, 9)
    assert dataset[0].y.size() == (1,)
    assert dataset[0].adj.size() == (4, 38, 38)
    assert dataset[0].bfs_perm_origin.size() == (38,)
    assert dataset[0].num_atom.size() == (1,)

    dataset = ZINC250k(root, one_shot=True, prop_name='penalized_logp')
    
    assert len(dataset) == 249455
    assert dataset.__repr__() == 'zinc250k_property(249455)'
    assert len(dataset.get_split_idx()) == 2

    assert len(dataset[0]) == 5
    assert dataset[0].x.size() == (10, 38)
    assert dataset[0].y.size() == (1,)
    assert dataset[0].adj.size() == (4, 38, 38)
    assert dataset[0].num_atom.size() == (1,)

    shutil.rmtree(root)

if __name__ == '__main__':
    test_zinc250k()