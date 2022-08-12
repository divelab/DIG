from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor



def compute_micro_f1(logits: Tensor, y: Tensor,
                     mask: Optional[Tensor] = None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.


def gen_masks(y: Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = int(y.max()) + 1

    train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        train_mask.scatter_(0, train_idx, True)
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        val_mask.scatter_(0, val_idx, True)

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask


def dropout(adj_t: SparseTensor, p: float, training: bool = True):
    if not training or p == 0.:
        return adj_t

    if adj_t.storage.value() is not None:
        value = F.dropout(adj_t.storage.value(), p=p)
        adj_t = adj_t.set_value(value, layout='coo')
    else:
        mask = torch.rand(adj_t.nnz(), device=adj_t.storage.row().device) > p
        adj_t = adj_t.masked_select_nnz(mask, layout='coo')

    return adj_t
