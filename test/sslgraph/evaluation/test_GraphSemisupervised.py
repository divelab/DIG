import shutil
from dig.sslgraph.evaluation import GraphSemisupervised
from dig.sslgraph.utils import Encoder
from dig.sslgraph.dataset import get_dataset
from dig.sslgraph.method import GraphCL
from dig.sslgraph.utils import setup_seed
from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask
from dig.sslgraph.method import Contrastive

class SSLModel(Contrastive):
    def __init__(self, z_dim, mask_ratio, **kwargs):

        objective = "JSE"
        proj="MLP"
        mask_i = NodeAttrMask(mask_ratio=mask_ratio)
        mask_j = NodeAttrMask(mask_ratio=mask_ratio)
        views_fn = [mask_i, mask_j]

        super(SSLModel, self).__init__(objective=objective,
                                    views_fn=views_fn,
                                    z_dim=z_dim,
                                    proj=proj,
                                    node_level=False,
                                    **kwargs)

    def train(self, encoder, data_loader, optimizer, epochs, per_epoch_out=False):
        for enc, proj in super(SSLModel, self).train(encoder, data_loader,
                                                    optimizer, epochs, per_epoch_out):
            yield enc

class SSLModel_2(Contrastive):
    def __init__(self, z_dim, mask_ratio, **kwargs):

        objective = "JSE"
        proj="MLP"
        mask_i = NodeAttrMask(mode='partial', mask_ratio=mask_ratio)
        mask_j = NodeAttrMask(mode='onehot', mask_ratio=mask_ratio)
        views_fn = [mask_i, mask_j]

        super(SSLModel_2, self).__init__(objective=objective,
                                    views_fn=views_fn,
                                    z_dim=z_dim,
                                    proj=proj,
                                    neg_by_crpt=True,
                                    choice_model='best',
                                    node_level=False,
                                    **kwargs)

    def train(self, encoder, data_loader, optimizer, epochs, per_epoch_out=False):
        for enc, proj in super(SSLModel_2, self).train(encoder, data_loader,
                                                    optimizer, epochs, per_epoch_out):
            yield enc


def test_GraphSemisupervised():
    root = './dataset'
    dataset, dataset_pretrain = get_dataset('NCI1', task='semisupervised', root=root)
    feat_dim = dataset[0].x.shape[1]
    embed_dim = 8

    encoder = Encoder(feat_dim, embed_dim, n_layers=2, gnn='resgcn')
    graphcl = GraphCL(embed_dim, aug_1='dropN', aug_2='permE')

    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01, p_epoch = 1, f_epoch = 1)

    test_mean, test_std = evaluator.evaluate(learning_model=graphcl, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    encoder = Encoder(feat_dim, embed_dim, n_layers=2, gnn='resgcn')
    graphcl = GraphCL(embed_dim, aug_1='subgraph', aug_2='maskN')

    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01, p_epoch = 1, f_epoch = 1)

    test_mean, test_std = evaluator.evaluate(learning_model=graphcl, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    encoder = Encoder(feat_dim, embed_dim, n_layers=2, gnn='resgcn')
    graphcl = GraphCL(embed_dim, aug_1='random3', aug_2='random4')

    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01, p_epoch = 1, f_epoch = 1)

    test_mean, test_std = evaluator.evaluate(learning_model=graphcl, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    encoder = Encoder(feat_dim, embed_dim, n_layers=2, gnn='resgcn')
    graphcl = GraphCL(embed_dim, aug_1='random2', aug_2='random4')

    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01, p_epoch = 1, f_epoch = 1)

    test_mean, test_std = evaluator.evaluate(learning_model=graphcl, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    encoder = Encoder(feat_dim, embed_dim, n_layers=2, gnn='resgcn')
    ssl_model = SSLModel(z_dim=embed_dim, mask_ratio=0.1)

    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01, p_epoch = 1, f_epoch = 1)

    test_mean, test_std = evaluator.evaluate(learning_model=ssl_model, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    encoder = Encoder(feat_dim, embed_dim, n_layers=2, gnn='resgcn')
    ssl_model_2 = SSLModel_2(z_dim=embed_dim, mask_ratio=0.1)

    evaluator = GraphSemisupervised(dataset, dataset_pretrain, label_rate=0.01, p_epoch = 1, f_epoch = 1)

    test_mean, test_std = evaluator.evaluate(learning_model=ssl_model_2, encoder=encoder)

    assert test_mean <= 1.0 and test_mean >= 0.0
    assert test_std is not None

    shutil.rmtree(root)

if __name__ == '__main__':
    setup_seed(0)
    test_GraphSemisupervised()
