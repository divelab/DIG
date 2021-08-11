import shutil
from dig.sslgraph.dataset import get_node_dataset


def test_get_node_dataset():
    root = './dataset/Planetoid'
    # Cora
    dataset = get_node_dataset('Cora', norm_feat=False, root=root)
    assert len(dataset) == 1
    assert dataset.num_features == 1433 

    assert dataset[0].x.size() == (2708, 1433)
    assert dataset[0].y.size() == (2708,)
    assert dataset[0].edge_index.size() == (2, 10556)

    shutil.rmtree(root)

    # CiteSeer
    dataset = get_node_dataset('CiteSeer', norm_feat=False, root=root)
    assert len(dataset) == 1
    assert dataset.num_features == 3703

    assert dataset[0].x.size() == (3327, 3703)
    assert dataset[0].y.size() == (3327,)
    assert dataset[0].edge_index.size() == (2, 9104)

    shutil.rmtree(root)

    # PubMed
    dataset = get_node_dataset('PubMed', norm_feat=False, root=root)
    assert len(dataset) == 1
    assert dataset.num_features == 500

    assert dataset[0].x.size() == (19717, 500)
    assert dataset[0].y.size() == (19717,)
    assert dataset[0].edge_index.size() == (2, 88648)

    shutil.rmtree(root)

if __name__ == '__main__':
    test_get_node_dataset()