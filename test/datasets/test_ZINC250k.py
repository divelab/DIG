from dig.ggraph.dataset import ggraph_dataset

def test_zinc250k():
    dataset = ggraph_dataset.ZINC250k(root='./dataset', prop_name='penalized_logp')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    data = next(iter(loader))
    assert len(data.y) == 32

