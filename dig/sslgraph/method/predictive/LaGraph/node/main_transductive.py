import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
from models import LogReg, LaGraphNetNode
from utils import process, pyg_data
import argparse
from loss import LaGraphLoss
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn import metrics
import sklearn.metrics as sklm
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.metrics import f1_score
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph


parser = argparse.ArgumentParser("LaGraph Node Task")
parser.add_argument('--dataset',          type=str,           default="Computers",                help='data')
parser.add_argument('--seed',             type=int,           default=1,                help='seed')
parser.add_argument('--mratio', default=0.05, type=float, help='Mask ratio')
parser.add_argument('--mstd', default=0.5, type=float, help='Mask std dev')
parser.add_argument('--mmode', default='whole', type=str, help='Mask whole or partial node')
parser.add_argument('--alpha', default=2, type=float, help='Weight of loss_inv')
parser.add_argument('--enly', default=2, type=int, help='Number of encoder layers')
parser.add_argument('--dely', default=1, type=int, help='Number of decoder layers')
parser.add_argument('--decoder', default='mlp', type=str, help='Type of decoder')
parser.add_argument('--dim_eb', default=512, type=int, help='Dimension of embedding layer')
parser.add_argument('--hidden', default=512, type=int, help='Dimension of hidden layer')
parser.add_argument('--nb_epochs', default=2, type=int, help='Pretrain epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate of pretrain')
parser.add_argument('--interval', default=1, type=int)
parser.add_argument('--wd', default=0.0, type=float, help='Weight decay of LogReg')
parser.add_argument('--pwd', default=0.0, type=float, help='Weight decay of pretrain')
parser.add_argument('--reg_ep', default=100, type=int, help='LogReg epoch')
parser.add_argument('--feature', default=False, type=bool, help='Concatenate features to embeds')



args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sparse = True


orig_edge_index, orig_features, labels, idxs_train, idxs_val, idxs_test = pyg_data.load(args.dataset)
edge_index = orig_edge_index
adj = to_scipy_sparse_matrix(edge_index=edge_index)

nb_nodes = orig_features.shape[0]  # node number
ft_size = orig_features.shape[1]   # node features dim
nb_classes = torch.max(labels).item()+1

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
if sparse:
    orig_sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])

orig_features = torch.FloatTensor(orig_features[np.newaxis])
labels = torch.LongTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idxs_train)
idx_val = torch.LongTensor(idxs_val)
idx_test = torch.LongTensor(idxs_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LaGraphNetNode(dim_ft=ft_size, dim_hid=args.hidden, dim_eb=args.dim_eb, num_en_layers=args.enly, num_de_layers=args.dely, decoder=args.decoder)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.pwd)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    orig_features = orig_features.cuda()
    if sparse:
        orig_sp_adj = orig_sp_adj.cuda()
    else:
        adj = adj.cuda()

    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
best = 1e9
best_acc = 0

for epoch in range(args.nb_epochs):
    
    sp_adj = orig_sp_adj
    data = orig_features
    data.to(device)

    model.train()
    optimiser.zero_grad()

    embed_orig, enc_orig, dec_orig, _ = model(data, sp_adj if sparse else adj, sparse, args, aug=False)
    _, enc_aug, _, mask = model(deepcopy(data), sp_adj if sparse else adj, sparse, args, aug=True)
    loss, loss_rec, loss_inv = LaGraphLoss()(embed_orig, enc_orig[0], dec_orig, enc_aug[0], mask, args.alpha)

    print('Epoch: {0:0d}, Loss:[{1:.4f}]'.format(epoch, loss.item()))
    
    loss.backward()
    optimiser.step()

    if epoch % args.interval == 0:

        embeds = model.embed(orig_features, orig_sp_adj if sparse else adj, sparse)

        tot = torch.zeros(1)
        if torch.cuda.is_available():
            tot = tot.cuda()

        accs = []
        for i in range(20):

            if args.dataset not in ['flickr', 'reddit']:
                idx_train = idxs_train[i]
                idx_val = idxs_val[i]
                idx_test = idxs_test[i]
            train_embs = embeds[0, idx_train]
            val_embs = embeds[0, idx_val]
            test_embs = embeds[0, idx_test]
            if args.feature:
                train_fts = orig_features[0, idx_train]
                val_fts = orig_features[0, idx_val]
                test_fts = orig_features[0, idx_test]
                train_embs = torch.cat((train_embs, train_fts), 1)
                val_embs = torch.cat((val_embs, val_fts), 1)
                test_embs = torch.cat((test_embs, test_fts), 1)

            train_lbls = labels[0, idx_train]
            val_lbls = labels[0, idx_val]
            test_lbls = labels[0, idx_test]

            log = LogReg(train_embs.shape[1], nb_classes)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=args.wd)
            if torch.cuda.is_available():
                log.cuda()
                train_lbls.cuda()

            for _ in range(args.reg_ep):
                log.train()
                opt.zero_grad()
                logits = log(train_embs)
                loss = xent(logits, train_lbls)
                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            if args.dataset in ['flickr', 'reddit']:
                acc = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')  # f1 score
            else:
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc.item() * 100)
            # print('acc:[{:.4f}]'.format(acc))
            tot += acc

        if tot.item() / 20 > best_acc:
            best_acc = tot.item() / 20

        print('-' * 100)
        print('Average accuracy:[{:.4f}], Best accuracy:[{:.4f}]'.format(tot.item() / 20, best_acc))
        accs = np.array(accs)
        print('Mean:[{:.4f}]'.format(accs.mean()))
        print('Std :[{:.4f}]'.format(accs.std()))
        print('-' * 100)

        



