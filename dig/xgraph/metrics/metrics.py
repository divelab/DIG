"""
FileName: metrics.py
Description: 
Time: 2021/2/22 14:00
Project: DIG
Author: Shurui Gui
"""

import torch
import torch.nn as nn
from typing import List, Union
from torch import Tensor
import numpy as np
import cilog
from torch_geometric.nn import MessagePassing



def control_sparsity(mask, sparsity=None):
    r"""

    :param mask: mask that need to transform
    :param sparsity: sparsity we need to control i.e. 0.7, 0.5
    :return: transformed mask where top 1 - sparsity values are set to inf.
    """
    if sparsity is None:
        sparsity = 0.7

    # Not apply here, Please refer to specific explainers in other directories
    #
    # if data_args.model_level == 'node':
    #     assert self.hard_edge_mask is not None
    #     mask_indices = torch.where(self.hard_edge_mask)[0]
    #     sub_mask = mask[self.hard_edge_mask]
    #     mask_len = sub_mask.shape[0]
    #     _, sub_indices = torch.sort(sub_mask, descending=True)
    #     split_point = int((1 - sparsity) * mask_len)
    #     important_sub_indices = sub_indices[: split_point]
    #     important_indices = mask_indices[important_sub_indices]
    #     unimportant_sub_indices = sub_indices[split_point:]
    #     unimportant_indices = mask_indices[unimportant_sub_indices]
    #     trans_mask = mask.clone()
    #     trans_mask[:] = - float('inf')
    #     trans_mask[important_indices] = float('inf')
    # else:
    _, indices = torch.sort(mask, descending=True)
    mask_len = mask.shape[0]
    split_point = int((1 - sparsity) * mask_len)
    important_indices = indices[: split_point]
    unimportant_indices = indices[split_point:]
    trans_mask = mask.clone()
    trans_mask[important_indices] = float('inf')
    trans_mask[unimportant_indices] = - float('inf')

    return trans_mask


def fidelity(ori_probs: torch.Tensor, unimportant_probs: torch.Tensor) -> float:

    drop_probability = ori_probs - unimportant_probs

    return drop_probability.mean().item()


def fidelity_inv(ori_probs: torch.Tensor, important_probs: torch.Tensor) -> float:

    drop_probability = ori_probs - important_probs

    return drop_probability.mean().item()


class XCollector(object):

    def __init__(self, sparsity):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity = sparsity
        self.__fidelity, self.__fidelity_inv = None, None
        self.__score = None


    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__fidelity, self.__fidelity_inv = None, None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int) -> None:

        if self.__fidelity or self.__fidelity_inv:
            self.__fidelity, self.__fidelity_inv = None, None
            print(f'#W#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)

            self.__targets.append(label)
            self.masks.append(masks)


    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        else:

            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            self.__fidelity = fidelity(one_mask_preds, mask_out_preds)
            return self.__fidelity

    @property
    def fidelity_inv(self):
        if self.__fidelity_inv:
            return self.__fidelity_inv
        else:

            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            self.__fidelity_inv = fidelity_inv(one_mask_preds, masked_preds)
            return self.__fidelity_inv

    @property
    def sparsity(self):
        if self.__sparsity:
            return self.__sparsity
        else:
            raise ValueError(f'Please control and set your '
                             f'Sparsity when initializing this class instead of calculating it.')


class ExplanationProcessor(nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self.edge_mask = None
        self.model = model
        self.device = device
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                                 [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain__ = True
                module.__edge_mask__ = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain__ = False

    def eval_related_pred(self, x, edge_index, masks, **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx # graph level: 0, node level: node_idx

        related_preds = []

        for label, mask in enumerate(masks):
            # origin pred
            for edge_mask in self.edge_mask:
                edge_mask.data = float('inf') * torch.ones(mask.size(), device=self.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            for edge_mask in self.edge_mask:
                edge_mask.data = mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            for edge_mask in self.edge_mask:
                edge_mask.data = - mask
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            for edge_mask in self.edge_mask:
                edge_mask.data = - float('inf') * torch.ones(mask.size(), device=self.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # Store related predictions for further evaluation.
            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

            # Adding proper activation function to the models' outputs.
            related_preds[label] = {key: pred.softmax(0)[label].item()
                                    for key, pred in related_preds[label].items()}

        return related_preds

    def forward(self, data, masks, x_collector: XCollector, **kwargs):

        data.to(self.device)
        node_idx = kwargs.get('node_idx')
        y_idx = 0 if node_idx is None else node_idx

        assert not torch.isnan(data.y[y_idx].squeeze())

        self.num_edges = data.edge_index.shape[1]
        self.num_nodes = data.x.shape[0]

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(data.x, data.edge_index, masks, **kwargs)

        x_collector.collect_data(masks,
                                 related_preds,
                                 data.y[y_idx].squeeze().long().item())


# --- Explanation evaluation demo ---
def demo():
    import os
    from defi import ROOT_DIR
    import sys
    sys.path.append(os.path.abspath(os.path.join(ROOT_DIR, '..', 'DeepLIFT')))
    print(f"Add {os.path.abspath(os.path.join(ROOT_DIR, '..', 'DeepLIFT'))} as a system path.")

    cilog.create_logger(sub_print=True)

    from models import GCN_3l
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    from torch_geometric.utils.random import barabasi_albert_graph
    from torch_geometric.data import Data

    # --- Create a 3-layer GCN model ---
    model = GCN_3l().to(device=device)

    # --- Set the Sparsity to 0.5 ---
    sparsity = 0.5

    # --- Create data collector and explanation processor ---
    x_collector = XCollector(sparsity)
    x_processor = ExplanationProcessor(model=model, device=device)

    # --- Given a 2-class classification with 10 explanation ---
    num_classes = 2
    for _ in range(10):

        # --- Create random ten-node BA graph ---
        x = torch.ones((10, 1), dtype=torch.float)
        edge_index = barabasi_albert_graph(10, 3)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([1.])) # Assume that y is the ground-truth valuing 1

        # --- Create random explanation ---
        masks = [control_sparsity(torch.randn(edge_index.shape[1], device=device), sparsity) for _ in range(num_classes)]

        # --- Process the explanation including data collection ---
        x_processor(data, masks, x_collector)

    # --- Get the evaluation metric results from the data collector ---
    print(f'#I#Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

if __name__ == '__main__':
    demo()
