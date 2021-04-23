"""
FileName: explainers.py
Description: Explainable methods' set
Time: 2020/8/4 8:56
Project: GNN_benchmark
Author: Shurui Gui
"""
from typing import Any, Callable, List, Tuple, Union, Dict, Sequence

from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from benchmark.models.utils import subgraph, normalize
from torch.nn.functional import binary_cross_entropy as bceloss
from typing_extensions import Literal
from benchmark.kernel.utils import Metric
from benchmark.data.dataset import data_args
from benchmark.args import x_args
from rdkit import Chem
from matplotlib.axes import Axes
from matplotlib.patches import Path, PathPatch


import captum
import captum.attr as ca
from captum.attr._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.deep_lift import DeepLiftShap
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.attr._utils.common import (
    ExpansionTypes,
    _call_custom_attribution_func,
    _compute_conv_delta_and_format_attrs,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_attributions,
    _format_baseline,
    _format_callable_baseline,
    _format_input,
    _tensorize_baseline,
    _validate_input,
)
from captum.attr._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
import benchmark.models.gradient_utils as gu

from itertools import combinations
import numpy as np
from benchmark.models.models import GlobalMeanPool, GraphSequential, GNNPool
from benchmark.models.ext.deeplift.layer_deep_lift import LayerDeepLift, DeepLift
import shap
import time

EPS = 1e-15


class ExplainerBase(nn.Module):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.explain_graph = explain_graph
        self.molecule = molecule
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.hard_edge_mask = None

        self.num_edges = None
        self.num_nodes = None
        self.device = None
        self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)
        # self.edge_mask = torch.nn.Parameter(100 * torch.ones(E, requires_grad=True))

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def __num_hops__(self):
        if self.explain_graph:
            return -1
        else:
            return self.num_layers

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device


    def control_sparsity(self, mask, sparsity=None):
        r"""

        :param mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
        """
        if sparsity is None:
            sparsity = 0.7

        if data_args.model_level == 'node':
            assert self.hard_edge_mask is not None
            mask_indices = torch.where(self.hard_edge_mask)[0]
            sub_mask = mask[self.hard_edge_mask]
            mask_len = sub_mask.shape[0]
            _, sub_indices = torch.sort(sub_mask, descending=True)
            split_point = int((1 - sparsity) * mask_len)
            important_sub_indices = sub_indices[: split_point]
            important_indices = mask_indices[important_sub_indices]
            unimportant_sub_indices = sub_indices[split_point:]
            unimportant_indices = mask_indices[unimportant_sub_indices]
            trans_mask = mask.clone()
            trans_mask[:] = - float('inf')
            trans_mask[important_indices] = float('inf')
        else:
            _, indices = torch.sort(mask, descending=True)
            mask_len = mask.shape[0]
            split_point = int((1 - sparsity) * mask_len)
            important_indices = indices[: split_point]
            unimportant_indices = indices[split_point:]
            trans_mask = mask.clone()
            trans_mask[important_indices] = float('inf')
            trans_mask[unimportant_indices] = - float('inf')

        return trans_mask


    def visualize_graph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs) -> Tuple[Axes, nx.DiGraph]:
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=kwargs.get('num_nodes'))
        assert edge_mask.size(0) == edge_index.size(1)

        if self.molecule:
            atomic_num = torch.clone(y)

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        # --- temp ---
        edge_mask[edge_mask == float('inf')] = 1
        edge_mask[edge_mask == - float('inf')] = 0
        # ---

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if data_args.dataset_name == 'ba_lrp':
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset]

        if self.molecule:
            atom_colors = {6: '#8c69c5', 7: '#71bcf0', 8: '#aef5f1', 9: '#bdc499', 15: '#c22f72', 16: '#f3ea19',
                           17: '#bdc499', 35: '#cc7161'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]
        else:
            atom_colors = {0: '#8c69c5', 1: '#c56973', 2: '#a1c569', 3: '#69c5ba'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]


        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 250
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        # calculate Graph positions
        pos = nx.kamada_kawai_layout(G)
        ax = plt.gca()

        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    lw=max(data['att'], 0.5) * 2,
                    alpha=max(data['att'], 0.4),  # alpha control transparency
                    color='#e1442a',  # color control color
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.08",  # rad control angle
                ))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, **kwargs)
        # define node labels
        if self.molecule:
            if x_args.nolabel:
                node_labels = {n: f'{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            else:
                node_labels = {n: f'{n}:{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
        else:
            if not x_args.nolabel:
                nx.draw_networkx_labels(G, pos, **kwargs)

        return ax, G

    def visualize_walks(self, node_idx, edge_index, walks, edge_mask, y=None,
                        threshold=None, **kwargs) -> Tuple[Axes, nx.DiGraph]:
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=kwargs.get('num_nodes'))
        assert edge_mask.size(0) == self_loop_edge_index.size(1)

        if self.molecule:
            atomic_num = torch.clone(y)

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = subgraph(
            node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        # --- temp ---
        edge_mask[edge_mask == float('inf')] = 1
        edge_mask[edge_mask == - float('inf')] = 0
        # ---

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if data_args.dataset_name == 'ba_lrp':
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset]

        if self.molecule:
            atom_colors = {6: '#8c69c5', 7: '#71bcf0', 8: '#aef5f1', 9: '#bdc499', 15: '#c22f72', 16: '#f3ea19',
                           17: '#bdc499', 35: '#cc7161'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]
        else:
            atom_colors = {0: '#8c69c5', 1: '#c56973', 2: '#a1c569', 3: '#69c5ba'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 8
        kwargs['node_size'] = kwargs.get('node_size') or 200
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        # calculate Graph positions
        pos = nx.kamada_kawai_layout(G)
        ax = plt.gca()

        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                    lw=1.5,
                    alpha=0.5,  # alpha control transparency
                    color='grey',  # color control color
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0",  # rad control angle
                ))


        # --- try to draw a walk ---
        walks_ids = walks['ids']
        walks_score = walks['score']
        walks_node_list = []
        for i in range(walks_ids.shape[1]):
            if i == 0:
                walks_node_list.append(self_loop_edge_index[:, walks_ids[:, i].view(-1)].view(2, -1))
            else:
                walks_node_list.append(self_loop_edge_index[1, walks_ids[:, i].view(-1)].view(1, -1))
        walks_node_ids = torch.cat(walks_node_list, dim=0).T

        walks_mask = torch.zeros(walks_node_ids.shape, dtype=bool, device=self.device)
        for n in G.nodes():
            walks_mask = walks_mask | (walks_node_ids == n)
        walks_mask = walks_mask.sum(1) == walks_node_ids.shape[1]

        sub_walks_node_ids = walks_node_ids[walks_mask]
        sub_walks_score = walks_score[walks_mask]

        for i, walk in enumerate(sub_walks_node_ids):
            verts = [pos[n.item()] for n in walk]
            if walk.shape[0] == 3:
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            else:
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            if sub_walks_score[i] > 0:
                patch = PathPatch(path, facecolor='none', edgecolor='red', lw=1.5,#e1442a
                                  alpha=(sub_walks_score[i] / (sub_walks_score.max() * 2)).item())
            else:
                patch = PathPatch(path, facecolor='none', edgecolor='blue', lw=1.5,#18d66b
                                  alpha=(sub_walks_score[i] / (sub_walks_score.min() * 2)).item())
            ax.add_patch(patch)


        nx.draw_networkx_nodes(G, pos, node_color=node_colors, **kwargs)
        # define node labels
        if self.molecule:
            if x_args.nolabel:
                node_labels = {n: f'{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            else:
                node_labels = {n: f'{n}:{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
        else:
            if not x_args.nolabel:
                nx.draw_networkx_labels(G, pos, **kwargs)

        return ax, G

    def eval_related_pred(self, x, edge_index, edge_masks, **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx  # graph level: 0, node level: node_idx
        related_preds = []

        for ex_label, edge_mask in enumerate(edge_masks):

            self.edge_mask.data = float('inf') * torch.ones(edge_mask.size(), device=data_args.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            self.edge_mask.data = edge_mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            self.edge_mask.data = - edge_mask  # keep Parameter's id
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            self.edge_mask.data = - float('inf') * torch.ones(edge_mask.size(), device=data_args.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

            # Adding proper activation function to the models' outputs.
            if 'cs' in Metric.cur_task:
                related_preds[ex_label] = {key: pred.softmax(0)[ex_label].item()
                                        for key, pred in related_preds[ex_label].items()}

        return related_preds


class GNNExplainer(ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, explain_graph=False, molecule=False):
        super(GNNExplainer, self).__init__(model, epochs, lr, explain_graph, molecule)



    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = Metric.loss_func(raw_preds, x_label)
        else:
            loss = Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs
                          ) -> None:

        # initialize a mask
        self.to(x.device)
        self.mask_features = mask_features

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x
            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)
            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.edge_mask.data

    def forward(self, x, edge_index, mask_features=False,
                positive=True, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            data (Batch): batch from dataloader
            edge_index (LongTensor): The edge indices.
            pos_neg (Literal['pos', 'neg']) : get positive or negative mask
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.model.eval()

        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        # Assume the mask we will predict
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        # Calculate mask
        print('#D#Masks calculate...')
        edge_masks = []
        for ex_label in ex_labels:
            self.__clear_masks__()
            self.__set_masks__(x, self_loop_edge_index)
            edge_masks.append(self.control_sparsity(self.gnn_explainer_alg(x, edge_index, ex_label), sparsity=kwargs.get('sparsity')))
            # edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))


        print('#D#Predict...')

        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, edge_masks, **kwargs)


        self.__clear_masks__()

        return None, edge_masks, related_preds




    def __repr__(self):
        return f'{self.__class__.__name__}()'


