from dig.sslgraph.evaluation import GraphSemisupervised
from dig.sslgraph.utils import Encoder
from dig.sslgraph.dataset import get_dataset
from dig.sslgraph.method import GraphCL
import shutil

def test_GraphSemisupervised():
    root = './dataset'
    dataset, dataset_pretrain = get_dataset('NCI1', task='semisupervised', root=root)
    feat_dim = dataset[0].x.shape[1]
    embed_dim = 128

    encoder = Encoder(feat_dim, embed_dim, n_layers=3, gnn='resgcn')
    graphcl = GraphCL(embed_dim, aug_1='subgraph', aug_2='subgraph')

    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01, p_epoch = 1, f_epoch = 1)

    test_mean, test_std = evaluator.evaluate(learning_model=graphcl, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    shutil.rmtree(root)

if __name__ == '__main__':
    test_GraphSemisupervised()
