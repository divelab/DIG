import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Reddit, Reddit2
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
from torch import Tensor

from logger import Logger

parser = argparse.ArgumentParser(description='reddit2 NeighborSampler')
parser.add_argument('--device', type=int, default=6)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--inference_batch_size', type=int, default=1024)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=110)
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--eval_start', type=int, default=90)

parser.add_argument('--alpha', type=float, default=0.1) # \alpha * old + (1 - \alpha) * new

parser.add_argument('--norm', default=False, action='store_true')
parser.add_argument('--train_eps', default=False, action='store_true')

parser.add_argument('--neighbor_size', type=int, default=2)
parser.add_argument('--neighbor_size_list', type=int, action='append', default=[])

parser.add_argument('--runs', type=int, default=5)

args = parser.parse_args()
print(args)

if len(args.neighbor_size_list) == 0:
    neighbor_size=args.num_layers * [args.neighbor_size]
else:
    neighbor_size = args.neighbor_size_list
assert len(neighbor_size)==args.num_layers
print('neighbor_size:',neighbor_size)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset', 'reddit2')
dataset = Reddit2(path)
data = dataset[0]
# if args.normfeature:
#     data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
print('data size:',data.x.size(),data.edge_index.size())
edge_index = data.edge_index

from torch_geometric.data import NeighborSampler
train_loader = NeighborSampler(edge_index, node_idx=data.train_mask,
                            sizes=neighbor_size, batch_size=args.batch_size, shuffle=True,
                            num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=args.inference_batch_size, shuffle=False,
                                  num_workers=12)


class Conv(MessagePassing):
    def __init__(self, eps=0., train_eps=False):
        super(Conv, self).__init__(aggr = 'mean')
    #     self.initial_eps = eps
    #     if train_eps:
    #         self.eps = torch.nn.Parameter(torch.Tensor([eps]))
    #     else:
    #         self.register_buffer('eps', torch.Tensor([eps]))
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, size=None):
        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, size=size)
        # out += (1 + self.eps) * x[1]

        return out

    def message(self, x_j):
        return x_j



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0., normalize=False, train_eps=False):
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        self.train_eps = train_eps

        self.convs = torch.nn.ModuleList()
        self.lins_l = torch.nn.ModuleList()
        self.lins_r = torch.nn.ModuleList()

        self.convs.append(Conv(eps=0., train_eps=self.train_eps))
        self.lins_l.append(Linear(in_channels, hidden_channels, bias=True))
        self.lins_r.append(Linear(in_channels, hidden_channels, bias=False))
        for _ in range(self.num_layers - 2):
            self.convs.append(Conv(eps=0., train_eps=self.train_eps))
            self.lins_l.append(Linear(hidden_channels, hidden_channels, bias=True))
            self.lins_r.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.convs.append(Conv(eps=0., train_eps=self.train_eps))
        self.lins_l.append(Linear(hidden_channels, out_channels, bias=True))
        self.lins_r.append(Linear(hidden_channels, out_channels, bias=False))

    def reset_parameters(self):
        for lin_l in self.lins_l:
            lin_l.reset_parameters()
        for lin_r in self.lins_r:
            lin_r.reset_parameters()

    def forward(self, x, adjs, n_id):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index) # results before MLPs
            ####################################################################################
            ### moving average
            ####################################################################################
            if args.alpha > 0:
                x_target_history = node_emb_dict[i][n_id][:size[1]]
                x = (1 - args.alpha) * x + args.alpha * x_target_history
                node_emb_dict[i][n_id[:size[1]]] = x.detach().clone()
            if self.normalize:
                x = F.normalize(x, p=2., dim=-1)
            x = self.lins_l[i](x) + self.lins_r[i](x_target)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, disable=True)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if self.normalize:
                    x = F.normalize(x, p=2., dim=-1)
                x = self.lins_l[i](x) + self.lins_r[i](x_target)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


model = GNN(dataset.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout, normalize=args.norm, train_eps=args.train_eps)
model = model.to(device)
logger = Logger(args.runs, args)

x = data.x.to(device)
y = data.y.squeeze().to(device)

if args.alpha > 0:
    #############################################################
    ###### extra memory
    #############################################################
    layer_emb_sizes=[dataset.num_features]
    for i in range(args.num_layers-1):
        layer_emb_sizes.append(args.hidden_channels)
    print('layer_emb_sizes',layer_emb_sizes)

    node_size=len(y)
    node_emb_dict = {}
    for layer_index, emb_size in enumerate(layer_emb_sizes):
        if layer_index == 0:
            node_emb_dict[layer_index] = x.detach().clone().to(device)
        else:
            node_emb_dict[layer_index] = torch.zeros(node_size, emb_size).to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()),disable=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs, n_id)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results

for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    best_valid = 0.
    best_test = 0.
    best_valid_epoch = 0
    best_test_epoch = 0
    total_time = 0
    for epoch in range(1, args.epochs+1):
        t_start = time.perf_counter()
        loss, acc = train(epoch)
        t_end = time.perf_counter()
        total_time += t_end - t_start
        print('time:',t_end-t_start)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        if epoch > args.eval_start and epoch % args.eval_steps == 0:
            result = test()
            logger.add_result(run,result)
            train_acc, val_acc, test_acc = result
            if val_acc > best_valid:
                best_valid = val_acc
                best_valid_epoch = epoch
            if test_acc > best_test:
                best_test = test_acc
                best_test_epoch = epoch
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * val_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}% '
                  f'  |  Best Valid: {100 * best_valid:.2f}%, '
                  f'Best Valid epoch: {best_valid_epoch:02d} '
                  f'  |  Best Test: {100 * best_test:.2f}%, '
                  f'Best Test epoch: {best_test_epoch:02d}')
        scheduler.step()
    print('average training time per epoch:', total_time/args.epochs)
    logger.add_result(run,result)
    logger.print_statistics(run)
logger.print_statistics()