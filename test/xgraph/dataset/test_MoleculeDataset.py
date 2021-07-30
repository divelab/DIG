from dig.xgraph.dataset import MoleculeDataset
import shutil


def test_MoleculeDataset():
    root = 'datasets'
    dataset_names = ['mutag', 'bbbp', 'Tox21', 'bace']
    dataset_length = [188, 2039, 7831, 1513]
    dataset_x_shape = [(3371, 7), (49068, 9), (145459, 9), (51577, 9)]
    dataset_edge_index_shape = [(2, 7442), (2, 105842), (2, 302190), (2, 111536)]
    dataset_y_shape = [(188, ), (2039, 1), (7831, 12), (1513, 1)]

    first_data_x_shape = [(17, 7), (20, 9), (16, 9), (32, 9)]
    first_data_edge_index_shape = [(2, 38), (2, 40), (2, 34), (2, 70)]
    first_data_y_shape = [(1, ), (1, 1), (1, 12), (1, 1)]

    for dataset_idx, name in enumerate(dataset_names):
        dataset = MoleculeDataset(root, name)

        assert len(dataset) == dataset_length[dataset_idx]
        assert dataset.data.x.size() == dataset_x_shape[dataset_idx]
        assert dataset.data.edge_index.size() == dataset_edge_index_shape[dataset_idx]
        assert dataset.data.y.size() == dataset_y_shape[dataset_idx]

        data = dataset[0]
        assert data.x.size() == first_data_x_shape[dataset_idx]
        assert data.edge_index.size() == first_data_edge_index_shape[dataset_idx]
        assert data.y.size() == first_data_y_shape[dataset_idx]

    shutil.rmtree(root)


if __name__ == '__main__':
    test_MoleculeDataset()
