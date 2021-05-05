from dig.ggraph.dataset import ggraph_dataset

dataset = ggraph_dataset.ZINC250k(root='./dataset', prop_name='penalized_logp')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
data = next(iter(loader))
assert data.y == 32
#Batch(adj=[128, 38, 38], batch=[1216], bfs_perm_origin=[1216], num_atom=[32], ptr=[33], smile=[32], x=[1216, 9], y=[32])
