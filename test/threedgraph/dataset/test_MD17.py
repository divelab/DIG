from dig.threedgraph.dataset import MD17
import shutil

def test_MD17():
    root = './dataset'

    dataset = MD17(root=root, name='aspirin')
    assert len(dataset) == 211762
    assert dataset.__repr__() == 'MD17(211762)'
    assert len(dataset[0]) == 4
    assert dataset[0].z.size() == (21,)
    assert dataset[0].pos.size() == (21, 3)
    assert dataset[0].y.size() == (1,)
    assert dataset[0].force.size() == (21, 3)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
    assert split_idx['train'][0] == 118875
    assert split_idx['valid'][0] == 5044
    assert split_idx['test'][0] == 44424

#     dataset = MD17(root=root, name='benzene_old')
#     assert len(dataset) == 627983
#     assert dataset.__repr__() == 'MD17(627983)'
#     assert len(dataset[0]) == 4
#     assert dataset[0].z.size() == (12,)
#     assert dataset[0].pos.size() == (12, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].force.size() == (12, 3)
#     split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
#     assert split_idx['train'][0] == 401788
#     assert split_idx['valid'][0] == 32285
#     assert split_idx['test'][0] == 145596

#     dataset = MD17(root=root, name='ethanol')
#     assert len(dataset) == 555092
#     assert dataset.__repr__() == 'MD17(555092)'
#     assert len(dataset[0]) == 4
#     assert dataset[0].z.size() == (9,)
#     assert dataset[0].pos.size() == (9, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].force.size() == (9, 3)
#     split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
#     assert split_idx['train'][0] == 198217
#     assert split_idx['valid'][0] == 34712
#     assert split_idx['test'][0] == 144219

#     dataset = MD17(root=root, name='malonaldehyde')
#     assert len(dataset) == 993237
#     assert dataset.__repr__() == 'MD17(993237)'
#     assert len(dataset[0]) == 4
#     assert dataset[0].z.size() == (9,)
#     assert dataset[0].pos.size() == (9, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].force.size() == (9, 3)
#     split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
#     assert split_idx['train'][0] == 501619
#     assert split_idx['valid'][0] == 755114
#     assert split_idx['test'][0] == 866595

#     dataset = MD17(root=root, name='naphthalene')
#     assert len(dataset) == 326250
#     assert dataset.__repr__() == 'MD17(326250)'
#     assert len(dataset[0]) == 4
#     assert dataset[0].z.size() == (18,)
#     assert dataset[0].pos.size() == (18, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].force.size() == (18, 3)
#     split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
#     assert split_idx['train'][0] == 104200
#     assert split_idx['valid'][0] == 119477
#     assert split_idx['test'][0] == 158389

#     dataset = MD17(root=root, name='salicylic')
#     assert len(dataset) == 320231
#     assert dataset.__repr__() == 'MD17(320231)'
#     assert len(dataset[0]) == 4
#     assert dataset[0].z.size() == (16,)
#     assert dataset[0].pos.size() == (16, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].force.size() == (16, 3)
#     split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
#     assert split_idx['train'][0] == 210643
#     assert split_idx['valid'][0] == 217058
#     assert split_idx['test'][0] == 73884

#     dataset = MD17(root=root, name='toluene')
#     assert len(dataset) == 442790
#     assert dataset.__repr__() == 'MD17(442790)'
#     assert len(dataset[0]) == 4
#     assert dataset[0].z.size() == (15,)
#     assert dataset[0].pos.size() == (15, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].force.size() == (15, 3)
#     split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
#     assert split_idx['train'][0] == 84247
#     assert split_idx['valid'][0] == 303283
#     assert split_idx['test'][0] == 148214

#     dataset = MD17(root=root, name='uracil')
#     assert len(dataset) == 133770
#     assert dataset.__repr__() == 'MD17(133770)'
#     assert len(dataset[0]) == 4
#     assert dataset[0].z.size() == (12,)
#     assert dataset[0].pos.size() == (12, 3)
#     assert dataset[0].y.size() == (1,)
#     assert dataset[0].force.size() == (12, 3)
#     split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
#     assert split_idx['train'][0] == 37527
#     assert split_idx['valid'][0] == 54674
#     assert split_idx['test'][0] == 40298

    shutil.rmtree(root)
