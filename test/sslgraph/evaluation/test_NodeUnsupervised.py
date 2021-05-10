from dig.sslgraph.evaluation import NodeUnsupervised
from dig.sslgraph.dataset import get_node_dataset
from dig.sslgraph.utils import Encoder
from dig.sslgraph.method import GRACE
import shutil

def test_NodeUnsupervised():
    root = './dataset'
    dataset = get_node_dataset('cora', root=root)
    embed_dim = 128
    encoder = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, 
                    n_layers=2, gnn='gcn', node_level=True, graph_level=False)
    grace = GRACE(dim=embed_dim, dropE_rate_1=0.2, dropE_rate_2=0.4, 
                maskN_rate_1=0.3, maskN_rate_2=0.4, tau=0.4)
    
    evaluator = NodeUnsupervised(dataset, log_interval=1)
    evaluator.setup_train_config(p_lr=0.0005, p_epoch=1, p_weight_decay=1e-5, comp_embed_on='cpu')
    test_mean = evaluator.evaluate(learning_model=grace, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0

    shutil.rmtree(root)

if __name__ == '__main__':
    test_NodeUnsupervised()
