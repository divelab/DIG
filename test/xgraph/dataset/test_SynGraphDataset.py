from dig.xgraph.dataset import SynGraphDataset
import shutil


def test_SynGraphDataset():
    root = 'datasets'
    dataset_names = ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle', 'ba_2motifs']
    dataset_length = [1, 1, 1, 1, 1000]
    dataset_x_shape = [(700, 10), (1400, 10), (1231, 10), (871, 10), (25000, 10)]
    dataset_edge_index_shape = [(2, 4110), (2, 8920), (2, 3130), (2, 1942), (2, 50960)]
    dataset_y_shape = [(700, ), (1400, ), (1231, ), (871, ), (1000, )]

    for dataset_idx, name in enumerate(dataset_names):
        dataset = SynGraphDataset(root, name)

        assert len(dataset) == dataset_length[dataset_idx]
        assert dataset.data.x.size() == dataset_x_shape[dataset_idx]
        assert dataset.data.edge_index.size() == dataset_edge_index_shape[dataset_idx]
        assert dataset.data.y.size() == dataset_y_shape[dataset_idx]

        if name == 'ba_2motifs':
            data = dataset[0]
            assert data.x.size() == (25, 10)
            assert data.edge_index.size() == (2, 50)
            assert data.y.size() == (1, )

    shutil.rmtree(root)


if __name__ == '__main__':
    test_SynGraphDataset()
