import torch
from dig.sslgraph.method.contrastive.objectives import NCE_loss, JSE_loss

def test_nce_more_view():
    zs_1 = torch.randn((32, 32))
    zs_n_1 = torch.randn((32, 32))
    zs_2 = torch.randn((32, 32))
    zs_n_2 = torch.randn((32, 32))
    zs_3 = torch.randn((32, 32))
    zs_n_3 = torch.randn((32, 32))
    sigma = torch.tensor([[0,1,0],[1,0,1],[1,0,1]])

    loss = NCE_loss(zs=[zs_1,zs_2,zs_3],zs_n=[zs_n_1,zs_n_2,zs_n_3],batch=True,sigma=sigma)
    assert loss is not None
    loss = NCE_loss(zs=[zs_1,zs_2,zs_3],sigma=sigma)
    assert loss is not None

    loss = JSE_loss(zs=[zs_1,zs_2,zs_3],zs_n=[zs_n_1,zs_n_2,zs_n_3],batch=torch.zeros(32).long(),sigma=sigma)
    assert loss is not None
    loss = JSE_loss(zs=[zs_1,zs_2,zs_3],sigma=sigma)
    assert loss is not None

if __name__ == '__main__':
    test_nce_more_view()