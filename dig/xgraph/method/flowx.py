r"""
This is an official implementation of `FlowX: Towards Explainable Graph Neural Networks via Message Flows
<https://arxiv.org/abs/2206.12987>`_.
"""

from itertools import combinations
from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

from .base_explainer import WalkBase
from ..models.utils import subgraph, gumbel_softmax


def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)


class FlowX(WalkBase):
    r"""
    An implementation of FlowX in
    `FlowX: Towards Explainable Graph Neural Networks via Message Flows <https://arxiv.org/abs/2206.12987>`_.
    Args:
        model (torch.nn.Module): The target model prepared to explain.
        epochs (int, optional): The training steps.
        lr (float, optional): The explainer training learning rate.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)
    .. note::
            For node classification model, the :attr:`explain_graph` flag is False.
    """
    coeffs = {
        'edge_size': 5e-4,
        'edge_ent': 1e-1
    }

    def __init__(self, model, epochs=500, lr=3e-1, explain_graph=False, molecule=False):
        # def __init__(self, model, epochs=500, lr=1e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.ns_iter = 30
        self.ns_per_iter = None
        self.fidelity_plus = True
        self.score_lr = 0e-5  # 2e-5
        # self.alpha = 0.5

        self.no_mask = False
        if self.no_mask:
            self.epochs = 1
            self.lr = 0
            self.score_lr = 0

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:
        r"""
        Run the explainer for a specific graph instance.
        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            **kwargs (dict):
                :obj:`node_idx` （int, list, tuple, torch.Tensor): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.
        :rtype: (None, list, list)
        .. note::
            (None, masks, related_predictions):
            masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.
        """

        super().forward(x, edge_index, **kwargs)

        # Explanation initial process
        self.model.eval()

        # Initial original prediction
        self.ori_logits_pred = self.model(x, edge_index).softmax(1)
        # print(f'#I#pred label: {torch.argmax(self.ori_logits_pred)}')

        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)

        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]

        import time
        start = time.time()
        self.time_list = []
        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        # Connect mask
        with self.connect_mask(self):
            iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count = \
                self.flow_shap(x, edge_index, edge_index_with_loop, walk_indices_list, **kwargs)

        walk_score_list = []
        for ex_label in ex_labels:
            # --- training ---
            self.train_mask(x,
                            edge_index,
                            ex_label,
                            walk_indices_list,
                            edge_index_with_loop,
                            iter_weighted_change_walks_list,
                            iter_changed_subsets_score_list,
                            walk_sample_count)

            walk_score_list.append(self.flow_mask.data)

        walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}
        # print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
        # print(f'#D#WalkScore total time: {time.time() - start}\n'
        #       f'predict time: {sum(self.time_list)}')

        # specify to edge with self-loop mask prediction
        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            mask = self.control_sparsity(mask, kwargs.get('sparsity')).sigmoid()
            masks.append(mask.detach())
        # print(f'#D#Edge mask predict total time: {time.time() - start}')

        # Connect mask
        with self.connect_mask(self):
            related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return walks, masks, related_preds

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            loss = cross_entropy_with_logit(raw_preds[self.node_idx].unsqueeze(0), x_label)

        if self.fidelity_plus:
            loss = - loss

        # Option 2: make it hard: higher std, and closer to Sparsity
        # loss = loss - 10 * torch.square(self.mask - 0.5).mean()

        # loss = loss + 1e-3 * torch.square(self.mask.sum() - self.mask.shape[0] * (1 - x_args.sparsity))
        # m = self.nec_suf_mask.sigmoid()
        # loss = loss + self.coeffs['edge_size'] * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def train_mask(self,
                   x: Tensor,
                   edge_index: Tensor,
                   ex_label: Tensor,
                   walk_indices_list,
                   edge_index_with_loop,
                   iter_weighted_change_walks_list,
                   iter_changed_subsets_score_list,
                   walk_sample_count,
                   t0=7.,
                   t1=0.5,
                   **kwargs
                   ) -> None:
        # initialize a mask
        self.to(x.device)

        self.nec_suf_mask = nn.Parameter(
            1e-1 * nn.init.uniform_(torch.empty((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device)))


        if self.no_mask:
            self.nec_suf_mask = nn.Parameter(
                100 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))
        self.iter_weighted_change_walks_list = nn.Parameter(iter_weighted_change_walks_list.clone().detach())

        # --- Training ---
        walk_plain_indices_list = walk_indices_list + \
                                  (edge_index_with_loop.shape[1]
                                   * torch.arange(self.num_layers, device=self.device)).repeat(
                                      walk_indices_list.shape[0], 1)

        self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                  for i in
                                                  range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                 dim=1).detach()

        # train to get the mask
        optimizer = torch.optim.Adam([{'params': self.nec_suf_mask}],
                                     # {'params': self.iter_weighted_change_walks_list, 'lr': self.score_lr}],
                                     lr=self.lr)
        # print('#I#begin this ex_label')
        for epoch in range(1, self.epochs + 1):

            masked_iter_weighted_change_walks_list = self.iter_weighted_change_walks_list * self.nec_suf_mask.sigmoid()

            walk_scores = (masked_iter_weighted_change_walks_list.unsqueeze(3).repeat(1, 1, 1,
                                                                                      iter_changed_subsets_score_list.shape[
                                                                                          2]) * iter_changed_subsets_score_list.unsqueeze(
                2)).sum(1).sum(0)
            # EPS will affect the stability of training
            EPS = 1e-18
            shap_flow_score = (walk_scores / (walk_sample_count.unsqueeze(1) + EPS))

            # --- score/mask transformer ---
            self.flow_mask = shap_flow_score[:, ex_label]

            # --- setting layer edge masks ---
            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0],
                                                                                      self.num_layers,
                                                                                      -1).sum(0)
            mask = self.layer_edge_mask.sum(0)
            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            climb = True
            if climb:
                # pass
                mask = mask ** 8
            else:
                end_epoch = 300
                temperature = float(t0 * ((t1 / t0) ** (epoch / end_epoch))) if epoch < end_epoch else t1
                mask = gumbel_softmax(mask, temperature, training=True)

            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)
            cur_sparsity = (mask < 0.5).sum().float() / mask.shape[0]
            # if cur_sparsity < x_args.sparsity:
            #     # --- early stop ---
            #     break
            # if epoch % 20 == 0:
            #     print(f'Epoch: {epoch} --- training mask Sparsity: {cur_sparsity}')
            if self.fidelity_plus:
                mask = 1 - mask  # Fidelity +
            self.mask = mask
            # isig_mask = torch.log(self.mask / (1 - self.mask + EPS) + EPS)

            # --- temp update non-leaf edge_mask
            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                temp_edge_mask.append(mask)

            # debug:
            with self.temp_mask(self, temp_edge_mask):
                raw_preds = self.model(x, edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            # if epoch % 20 == 0:
            #     print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return

    def flow_shap(self,
                  x,
                  edge_index,
                  edge_index_with_loop,
                  walk_indices_list,
                  **kwargs
                  ):
        r"""
        Flow shapley calculations.
        """
        # --- Kernel algorithm ---

        # --- row: walk index, column: total score
        walk_sample_count = torch.zeros(walk_indices_list.shape[0], dtype=torch.float, device=self.device)

        # General setting
        iter_weighted_change_walks_list = []
        iter_changed_subsets_score_list = []

        # Random sample ns_iter * ns_per_iter times
        for iter_idx in range(self.ns_iter):

            # --- random index ---
            unmask_pool = torch.cat([walk_indices_list[:, layer].unique() + layer * edge_index_with_loop.shape[1]
                                     for layer in range(self.num_layers)])
            self.ns_per_iter = unmask_pool.shape[0] if self.explain_graph or unmask_pool.shape[
                0] <= 100 else 100
            idx = torch.randperm(unmask_pool.nelement())
            unmask_pool = unmask_pool.view(-1)[idx].view(unmask_pool.size())

            mask_per_sub = unmask_pool.shape[0] // self.ns_per_iter
            weighted_change_walks_list = []
            last_eliminated_walks = torch.zeros(walk_indices_list.shape[0], dtype=torch.bool, device=self.device)
            layer_edge_mask_list = []
            for sub_idx in range(self.ns_per_iter):
                # --- sub random index ---
                mask_pool = unmask_pool[: mask_per_sub * (sub_idx + 1)]

                # --- obtain changed walk idx ---
                eliminated_layer_edges = unmask_pool[mask_per_sub * sub_idx: mask_per_sub * (sub_idx + 1)]
                walk_plain_indices_list = walk_indices_list + \
                                          (edge_index_with_loop.shape[1] * torch.arange(self.num_layers,
                                                                                        device=self.device)).repeat(
                                              walk_indices_list.shape[0], 1)
                eliminated_walks = torch.stack([walk_plain_indices_list == edge for edge in eliminated_layer_edges],
                                               dim=0).long().sum(0).sum(1).bool().long()
                weighted_changed_walks = eliminated_walks.clone().float()
                weighted_changed_walks[eliminated_walks == last_eliminated_walks] = 0.
                weighted_changed_walks /= (weighted_changed_walks > 1e-20).sum() + 1e-30
                weighted_change_walks_list.append(weighted_changed_walks)
                last_eliminated_walks = eliminated_walks

                # --- setting a subset mask ---
                layer_edge_masks = torch.ones((self.num_layers, edge_index_with_loop.shape[1]),
                                              device=self.device)
                layer_edge_masks.view(-1)[mask_pool] -= 2
                layer_edge_mask_list.append(layer_edge_masks)

            weighted_change_walks_list = torch.stack(weighted_change_walks_list, dim=0)
            iter_weighted_change_walks_list.append(weighted_change_walks_list.detach())
            layer_edge_mask_list = torch.stack(layer_edge_mask_list, dim=0) * float('inf')

            # --- compute subsets' outputs of current iteration ---
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                self.edge_mask[layer_idx].data = torch.cat(
                    [layer_edge_mask_list[:, layer_idx, :self.num_edges].reshape(-1),
                     layer_edge_mask_list[:, layer_idx, self.num_edges:].reshape(-1)]).sigmoid()

            batch = self.batch_input(x, edge_index, self.ns_per_iter)
            subsets_output = self.model(data=batch).softmax(1).detach()
            if not self.explain_graph:
                subsets_output = subsets_output.view(self.ns_per_iter, -1, kwargs.get('num_classes'))[:, self.node_idx]
                last_subsets_output = torch.cat(
                    [self.ori_logits_pred[self.node_idx].unsqueeze(0), subsets_output.clone()[:-1]], dim=0)
            else:
                last_subsets_output = torch.cat([self.ori_logits_pred, subsets_output.clone()[:-1]], dim=0)

            changed_subsets_score_list = (last_subsets_output - subsets_output).detach()
            iter_changed_subsets_score_list.append(changed_subsets_score_list)

            walk_sample_count += (weighted_change_walks_list > 1e-30).float().sum(0)

        # iter x subset_idx x flow_idx
        iter_weighted_change_walks_list = torch.stack(iter_weighted_change_walks_list, dim=0)
        iter_changed_subsets_score_list = torch.stack(iter_changed_subsets_score_list, dim=0)

        return iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count
