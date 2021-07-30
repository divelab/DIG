from dig.xgraph.dataset import BA_LRP
import shutil


def test_BA_LRP():
    root = 'datasets'
    dataset = BA_LRP(root)
    gen_data1 = BA_LRP.gen_class1()
    gen_data2 = BA_LRP.gen_class2()

    assert len(dataset) == 20000
    assert dataset.data.edge_index.size() == (2, 840000)
    assert dataset.data.x.size() == (400000, 1)
    assert dataset.data.y.size() == (20000, 1)

    data = dataset[0]
    assert data.edge_index.size() == (2, 38)
    assert data.x.size() == (20, 1)
    assert data.y.size() == (1, 1)
    assert gen_data1.edge_index.size() == (2, 38)
    assert gen_data1.x.size() == (20, 1)
    assert gen_data1.y.size() == (19, 1)
    assert gen_data2.edge_index.size() == (2, 46)
    assert gen_data2.x.size() == (20, 1)
    assert gen_data2.y.size() == (23, 1)

    shutil.rmtree(root)


if __name__ == '__main__':
    test_BA_LRP()
