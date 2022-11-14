"""
Modified from the code "https://github.com/rusty1s/pyg_autoscale/blob/master/large_benchmark/main.py"
"""

import time
import hydra
from omegaconf import OmegaConf

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from dig.lsgraph.dataset import get_data, SubgraphLoader, EvalSubgraphLoader
from dig.lsgraph.method.GraphFMOB.metis import metis, permute
import dig.lsgraph.method.GraphFMOB.models as models
from dig.lsgraph.method.GraphFMOB.utils import compute_micro_f1, dropout

from torch import Tensor
from typing import Optional
from torch_geometric.typing import Adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

import logging
logger = logging.getLogger()


def random_walk_norm(edge_index: Adj,
                     edge_weight: Optional[Tensor] = None,
                     num_nodes: Optional[int] = None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if adj_t.storage.value() is None:
            adj_t = adj_t.set_value(torch.ones(adj_t.nnz()).to(adj_t.device()), layout='coo')
        deg = adj_t.sum(dim=1)
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        adj_t = adj_t.set_value(adj_t.storage.value() * deg_inv[adj_t.storage.row()],
                                layout='coo')

        return adj_t

    elif isinstance(edge_index, Tensor):
        row = edge_index[0]
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        deg = scatter_add(edge_weight, index=row, dim=0, dim_size=num_nodes)
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
        return edge_index, edge_weight


def mini_train(model, loader, criterion, optimizer, max_steps, grad_norm=None,
               edge_dropout=0.0):
    model.train()

    total_loss = total_examples = 0
    for i, (batch, batch_size, *args) in enumerate(loader):
        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:batch_size].to(model.device)
        train_mask = batch.train_mask[:batch_size].to(model.device)

        if train_mask.sum() == 0:
            continue

        # We make use of edge dropout on ogbn-products to avoid overfitting.
        adj_t = dropout(adj_t, p=edge_dropout)

        optimizer.zero_grad()
        # cut the args with only n_id, which makes the call to be _async=False
        args = args[:1]
        out = model(x, adj_t, batch_size, *args)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())

        # We may abort after a fixed number of steps to refresh histories...
        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break

    return total_loss / total_examples


@torch.no_grad()
def full_test(model, data):
    model.eval()
    return model(data.x.to(model.device), data.adj_t.to(model.device)).cpu()


@torch.no_grad()
def mini_test(model, loader):
    model.eval()
    return model(loader=loader)


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    conf.model.params = conf.model.params[conf.dataset.name]
    params = conf.model.params
    logger.info(OmegaConf.to_yaml(conf))
    try:
        edge_dropout = params.edge_dropout
    except:
        edge_dropout = 0.0

    grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'

    t = time.perf_counter()
    logger.info('Loading data...')
    data, in_channels, out_channels = get_data(conf.root, conf.dataset.name)
    logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')
    perm, ptr = metis(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    if conf.model.loop:
        t = time.perf_counter()
        logger.info('Adding self-loops...')
        data.adj_t = data.adj_t.set_diag()
        logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')
    if conf.model.norm:
        t = time.perf_counter()
        logger.info('Normalizing data...')
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')

    if data.y.dim() == 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
                                  shuffle=True, num_workers=params.num_workers,
                                  persistent_workers=params.num_workers > 0)

    eval_data = data
    eval_loader = EvalSubgraphLoader(
        eval_data, ptr, batch_size=params['batch_size'])

    t = time.perf_counter()
    logger.info('Calculating buffer size...')
    # We reserve a much larger buffer size than what is actually needed for
    # training in order to perform efficient history accesses during inference.

    # set buffer_size to None, which makes the model._async=False
    # Here, we implemented without offset and count in the forward method to disable the model._async
    buffer_size = max([n_id.numel() for _, _, n_id, _, _ in eval_loader])
    logger.info(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')

    kwargs = {}
    if conf.model.name[:3] == 'PNA':
        kwargs['deg'] = data.adj_t.storage.rowcount()

    GNN = getattr(models, conf.model.name)
    # we want to mask the model._async=False
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=conf.model.params.pool_size,
        buffer_size=buffer_size,
        gamma=conf.model.params.gamma,
        **params.architecture,
        **kwargs,
    ).to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(),
             weight_decay=params.reg_weight_decay),
        dict(params=model.nonreg_modules.parameters(),
             weight_decay=params.nonreg_weight_decay)
    ], lr=params.lr)

    t = time.perf_counter()
    logger.info('Fill history...')
    mini_test(model, eval_loader)
    logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')

    best_val_acc = test_acc = 0
    for epoch in range(1, params.epochs + 1):
        loss = mini_train(model, train_loader, criterion, optimizer,
                          params.max_steps, grad_norm, edge_dropout)
        out = mini_test(model, eval_loader)
        train_acc = compute_micro_f1(out, data.y, data.train_mask)

        val_acc = compute_micro_f1(out, data.y, data.val_mask)
        tmp_test_acc = compute_micro_f1(out, data.y, data.test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch % conf.log_every == 0:
            logger.info(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')

    logger.info('=========================')
    logger.info(f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
