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
from sklearn.metrics import f1_score
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from tqdm import tqdm



parser = argparse.ArgumentParser("LaGraph Node Task (Inductive PPI)")
parser.add_argument('--dataset',          type=str,           default="ppi",                help='data')
parser.add_argument('--seed',             type=int,           default=1,                help='seed')
parser.add_argument('--mratio', default=0.05, type=float, help='Mask ratio')
parser.add_argument('--mstd', default=0.5, type=float, help='Mask std dev')
parser.add_argument('--mmode', default='whole', type=str, help='Mask whole or partial node')
parser.add_argument('--alpha', default=2, type=float, help='Weight of loss_inv')
parser.add_argument('--enly', default=2, type=int, help='Number of encoder layers')
parser.add_argument('--dely', default=2, type=int, help='Number of decoder layers')
parser.add_argument('--decoder', default='mlp', type=str, help='Type of decoder')
parser.add_argument('--dim_eb', default=512, type=int, help='Weight of loss_inv')
parser.add_argument('--hidden', default=512, type=int, help='Weight of loss_inv')
parser.add_argument('--nb_epochs', default=2, type=int, help='Pretrain epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--interval', default=1, type=int)
parser.add_argument('--wd', default=0.0, type=float, help='Weight decay of LogReg')
parser.add_argument('--pwd', default=0.0, type=float, help='Weight decay of pretrain')
parser.add_argument('--reg_ep', default=100, type=int, help='LogReg epoch')
parser.add_argument('--feature', default=False, type=bool, help='Concatenate features to embeds')


args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sparse = True

train_edge_index, train_feat, train_labels, test_edge_index, test_feat, test_labels  = pyg_data.load(args.dataset)

adj_train = to_scipy_sparse_matrix(edge_index=train_edge_index)
adj_test = to_scipy_sparse_matrix(edge_index=test_edge_index)
sp_adj_train = train_edge_index
sp_adj_test = test_edge_index

ft_size = train_feat.shape[1]   # node features dim
nb_classes = 121

features_train = torch.FloatTensor(train_feat[np.newaxis])
labels_train = torch.FloatTensor(train_labels[np.newaxis])[0]
features_test = torch.FloatTensor(test_feat[np.newaxis])
labels_test = torch.FloatTensor(test_labels[np.newaxis])[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LaGraphNetNode(dim_ft=ft_size, dim_hid=args.hidden, dim_eb=args.dim_eb, num_en_layers=args.enly,
                       num_de_layers=args.dely, decoder=args.decoder, conv='pyg_gcn')

optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.pwd)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features_train = features_train.cuda()
    features_test = features_test.cuda()
    if sparse:
        sp_adj_train = sp_adj_train.cuda()
        sp_adj_test = sp_adj_test.cuda()
    else:
        adj = adj.cuda()

    labels_train = labels_train.cuda()
    labels_test = labels_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
best = 1e9
best_acc, best_std = 0, 0
best_epoch = 0

for epoch in tqdm(range(1, args.nb_epochs+1), total=args.nb_epochs, desc="Epoch: ", position=0, leave=True):
    data = features_train

    model.train()
    optimiser.zero_grad()

    embed_orig, enc_orig, dec_orig, _ = model(data, sp_adj_train if sparse else adj, sparse, args, aug=False)
    _, enc_aug, _, mask = model(deepcopy(data), sp_adj_train if sparse else adj, sparse, args, aug=True)
    loss, loss_rec, loss_inv = LaGraphLoss()(embed_orig, enc_orig[0], dec_orig, enc_aug[0], mask, args.alpha)

#     print('Epoch: {0:0d}, Loss:[{1:.4f}]'.format(epoch, loss.item()))

    loss.backward()
    optimiser.step()

    if epoch % args.interval == 0:
        
        train_embs = model.embed(features_train, sp_adj_train if sparse else adj, sparse)[0]
        test_embs = model.embed(features_test, sp_adj_test if sparse else adj, sparse)[0]

        accs = []
        for i in range(20):
            if args.feature:
                train_embs = torch.cat((train_embs, features_train[0]), 1)
                test_embs = torch.cat((test_embs, features_test[0]), 1)

            train_lbls = labels_train
            test_lbls = labels_test

            log = LogReg(train_embs.shape[1], nb_classes)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=args.wd)
            if torch.cuda.is_available():
                log.cuda()
                train_lbls.cuda()

            for _ in range(args.reg_ep):
                log.train()
                opt.zero_grad()
                logits = log(train_embs)
                loss = b_xent(logits, train_lbls)
                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = (logits > 0) * 1
            f1 = f1_score(test_lbls.view(-1).cpu(), preds.view(-1).cpu(), average='micro')
            accs.append(f1.item() * 100)
        
        accs = np.array(accs)

        if accs.mean().item() > best_acc:
            best_acc = accs.mean().item()
            best_std = accs.std().item()
            best_epoch = epoch

print('Best epoch: {0:d}, avg_f1: {1:.2f}, f1_std: {2:.2f}\n'.format(best_epoch, best_acc, best_std))

