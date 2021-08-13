import shutil
from dig.sslgraph.evaluation import GraphUnsupervised
from dig.sslgraph.utils import Encoder
from dig.sslgraph.dataset import get_dataset
from dig.sslgraph.method import GraphCL, InfoGraph, MVGRL


def test_GraphUnsupervised():
    root = './dataset'
    dataset = get_dataset('MUTAG', task='unsupervised', root=root)
    embed_dim = 8

    encoder = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, n_layers=2, gnn='gin', bn=True)
    graphcl = GraphCL(embed_dim*2, aug_1=None, aug_2='random2', tau=0.2)
    evaluator = GraphUnsupervised(dataset, log_interval=1, p_epoch=1)
    test_mean, test_std = evaluator.evaluate(learning_model=graphcl, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    embed_dim = 8
    encoder = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, n_layers=2, gnn='gin', node_level=True)
    infograph = InfoGraph(embed_dim*2, embed_dim)
    evaluator = GraphUnsupervised(dataset, log_interval=1, p_epoch=1)
    test_mean, test_std = evaluator.evaluate(learning_model=infograph, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    encoder_adj = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, 
                        n_layers=2, gnn='gcn', node_level=True, act='prelu')
    encoder_diff = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, 
                        n_layers=2, gnn='gcn', node_level=True, act='prelu', edge_weight=True)
    mvgrl = MVGRL(embed_dim*2, embed_dim)
    evaluator = GraphUnsupervised(dataset, log_interval=1, p_epoch=1)
    test_mean, test_std = evaluator.evaluate(learning_model=mvgrl, encoder=[encoder_adj, encoder_diff])

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None


    shutil.rmtree(root)
    
if __name__ == '__main__':
    test_GraphUnsupervised()
