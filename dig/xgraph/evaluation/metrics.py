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
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing



def control_sparsity(mask: torch.Tensor, sparsity: float=None):
    r"""
    Transform the mask where top 1 - sparsity values are set to inf.
    Args:
        mask (torch.Tensor): Mask that need to transform.
        sparsity (float): Sparsity we need to control i.e. 0.7, 0.5 (Default: :obj:`None`).
    :rtype: torch.Tensor
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
    r"""
    Return the Fidelity+ value according to collected data.

    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity+ computation.
        unimportant_probs (torch.Tensor): It is a vector providing probabilities without important features
            for Fidelity+ computation.

    :rtype: float

    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    drop_probability = ori_probs - unimportant_probs

    return drop_probability.mean().item()


def fidelity_inv(ori_probs: torch.Tensor, important_probs: torch.Tensor) -> float:
    r"""
    Return the Fidelity- value according to collected data.

    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity- computation.
        important_probs (torch.Tensor): It is a vector providing probabilities with only important features
            for Fidelity- computation.

    :rtype: float

    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    drop_probability = ori_probs - important_probs

    return drop_probability.mean().item()


class XCollector:
    r"""
    XCollector is a data collector which takes processed related prediction probabilities to calculate Fidelity+
    and Fidelity-.

    Args:
        sparsity (float): The Sparsity is use to transform the soft mask to a hard one.

    .. note::
        For more examples, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    """

    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = \
            {
                'zero': [],
                'masked': [],
                'maskout': [],
                'origin': [],
                'sparsity': [],
                'accuracy': [],
                'stability': []
             }, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity = sparsity
        self.__fidelity, self.__fidelity_inv, self.__accuracy, self.__stability = None, None, None, None
        self.__score = None

    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        r"""
        Clear class members.
        """
        self.__related_preds, self.__targets = \
            {
                'zero': [],
                'masked': [],
                'maskout': [],
                'origin': [],
                'sparsity': [],
                'accuracy': [],
                'stability': []
             }, []
        self.masks: Union[List, List[List[Tensor]]] = []
        self.__fidelity, self.__fidelity_inv, self.__accuracy, self.__stability = None, None, None, None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int = 0) -> None:
        r"""
        The function is used to collect related data. After collection, we can call fidelity, fidelity_inv, sparsity
        to calculate their values.

        Args:
            masks (list): It is a list of edge-level explanation for each class.
            related_preds (list): It is a list of dictionary for each class where each dictionary
            includes 4 type predicted probabilities and sparsity.
            label (int): The ground truth label. (default: 0)
        """

        if self.__fidelity is not None or self.__fidelity_inv is not None \
                or self.__accuracy is not None or self.__stability is not None:
            self.__fidelity, self.__fidelity_inv, self.__accuracy, self.__stability = None, None, None, None
            print(f'#W#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[0].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)

    @property
    def fidelity(self):
        r"""
        Return the Fidelity+ value according to collected data.

        .. note::
            Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
            <https://arxiv.org/abs/2012.15445>`_ for details.
        """
        if self.__fidelity is not None:
            return self.__fidelity
        else:
            if None in self.__related_preds['maskout'] or None in self.__related_preds['origin']:
                return None
            else:
                mask_out_preds, one_mask_preds = \
                    torch.tensor(self.__related_preds['maskout']), torch.tensor(self.__related_preds['origin'])

                self.__fidelity = fidelity(one_mask_preds, mask_out_preds)
                return self.__fidelity

    @property
    def fidelity_inv(self):
        r"""
        Return the Fidelity- value according to collected data.

        .. note::
            Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
            <https://arxiv.org/abs/2012.15445>`_ for details.
        """
        if self.__fidelity_inv is not None:
            return self.__fidelity_inv
        else:
            if None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
                return None
            else:
                masked_preds, one_mask_preds = \
                    torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

                self.__fidelity_inv = fidelity_inv(one_mask_preds, masked_preds)
                return self.__fidelity_inv

    @property
    def sparsity(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity is not None:
            return self.__sparsity
        else:
            if None in self.__related_preds['sparsity']:
                return None
            else:
                return torch.tensor(self.__related_preds['sparsity']).mean().item()

    @property
    def accuracy(self):
        r"""Return the accuracy for datasets with motif ground-truth"""
        if self.__accuracy is not None:
            return self.__accuracy
        else:
            if None in self.__related_preds['accuracy']:
                return torch.tensor([acc for acc in self.__related_preds['accuracy']
                                     if acc is not None]).mean().item()
            else:
                return torch.tensor(self.__related_preds['accuracy']).mean().item()

    @property
    def stability(self):
        r"""Return the accuracy for datasets with motif ground-truth"""
        if self.__stability is not None:
            return self.__stability
        else:
            if None in self.__related_preds['stability']:
                return torch.tensor([stability for stability in self.__related_preds['stability']
                                     if stability is not None]).mean().item()
            else:
                return torch.tensor(self.__related_preds['stability']).mean().item()


class ExplanationProcessor(nn.Module):
    r"""
    Explanation Processor is edge mask explanation processor which can handle sparsity control and use
    data collector automatically.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        device (torch.device): Specify running device: CPU or CUDA.

    """

    def __init__(self, model: nn.Module, device: torch.device):
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
                module._explain = True
                module.__edge_mask__ = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module._explain = False

    def eval_related_pred(self, x: torch.Tensor, edge_index: torch.Tensor, masks: List[torch.Tensor], **kwargs):

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

    def forward(self, data: Data, masks: List[torch.Tensor], x_collector: XCollector, **kwargs):
        r"""
        Please refer to the main function in `metric.py`.
        """

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
