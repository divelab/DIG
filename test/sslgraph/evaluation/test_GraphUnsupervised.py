from dig.sslgraph.evaluation import GraphUnsupervised
from dig.sslgraph.utils import Encoder
from dig.sslgraph.dataset import get_dataset
from dig.sslgraph.method import GraphCL
import shutil

def test_GraphUnsupervised():
    root = './dataset'
    dataset = get_dataset('MUTAG', task='unsupervised', root=root)
    embed_dim = 32
    encoder = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, n_layers=3, gnn='gin', bn=True)
    graphcl = GraphCL(embed_dim*3, aug_1=None, aug_2='random2', tau=0.2)
    evaluator = GraphUnsupervised(dataset, log_interval=1, p_epoch=5)
    test_mean, test_std = evaluator.evaluate(learning_model=graphcl, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    shutil.rmtree(root)
    
if __name__ == '__main__':
    test_GraphUnsupervised()