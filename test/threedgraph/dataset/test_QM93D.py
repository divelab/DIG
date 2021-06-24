from dig.threedgraph.dataset import QM93D
import shutil

def test_QM93D():
    root = './dataset'

    dataset = QM93D(root=root) 
    target='mu'
    dataset.data.y = dataset.data[target]

    assert len(dataset) == 130831
    assert dataset.__repr__() == 'QM93D(130831)'

    assert len(dataset[0]) == 15
    assert dataset[0].y.size() == (1,)
    assert dataset[0].z.size() == (5,)
    assert dataset[0].pos.size() == (5,3)
    assert dataset[0].Cv.size() == (1,)
    assert dataset[0].G.size() == (1,)
    assert dataset[0].H.size() == (1,)
    assert dataset[0].U.size() == (1,)
    assert dataset[0].U0.size() == (1,)
    assert dataset[0].alpha.size() == (1,)
    assert dataset[0].gap.size() == (1,)
    assert dataset[0].homo.size() == (1,)
    assert dataset[0].lumo.size() == (1,)
    assert dataset[0].mu.size() == (1,)
    assert dataset[0].r2.size() == (1,)
    assert dataset[0].zpve.size() == (1,)

    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=10000, seed=42)
    assert split_idx['train'][0] == 112526
    assert split_idx['valid'][0] == 120798
    assert split_idx['test'][0] == 107901

    shutil.rmtree(root)
