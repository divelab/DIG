import math
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils.loop import add_remaining_self_loops
from ..models.utils import subgraph
from .base_explainer import WalkBase

EPS = 1e-15


class GNN_GI(WalkBase):
    r"""
    An implementation of GNN-GI in
    `Higher-Order Explanations of Graph Neural Networks via Relevant Walks <https://arxiv.org/abs/2006.03589>`_.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)

    .. note:: For node classification model, the :attr:`explain_graph` flag is False.

    """

    def __init__(self, model: nn.Module, explain_graph: bool = False):
        super().__init__(model=model, explain_graph=explain_graph)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        r"""
        Run the explainer for a specific graph instance.

        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.

        :rtype: (dict, list, list)

        .. note::
            (walks, edge_masks, related_predictions):
            walks is a dictionary including walks' edge indices and corresponding explained scores;
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.

        """
        super().forward(x, edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_steps, fc_step = self.extract_step(x, edge_index, detach=False)
        labels = tuple(i for i in range(kwargs.get('num_classes')))

        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            if not node_idx.dim():
                node_idx = node_idx.reshape(-1)
            node_idx = node_idx.to(self.device)
            assert node_idx is not None
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

        if kwargs.get('walks'):
            walks = kwargs.pop('walks')

        else:
            def compute_walk_score(adjs, r, allow_edges, walk_idx=[]):
                if not adjs:
                    walk_indices.append(walk_idx)
                    walk_scores.append(r.detach())
                    return
                (grads,) = torch.autograd.grad(outputs=r, inputs=adjs[0], create_graph=True)
                for i in allow_edges:
                    allow_edges = torch.where(self_loop_edge_index[1] == self_loop_edge_index[0][i])[0].tolist()
                    new_r = grads[i] * adjs[0][i]
                    compute_walk_score(adjs[1:], new_r, allow_edges, [i] + walk_idx)

            walk_scores_tensor_list = [None for i in labels]
            for label in labels:
                if self.explain_graph:
                    f = torch.unbind(fc_step['output'][0, label].unsqueeze(0))
                    allow_edges = [i for i in range(self_loop_edge_index.shape[1])]
                else:
                    f = torch.unbind(fc_step['output'][node_idx, label].unsqueeze(0))
                    allow_edges = torch.where(self_loop_edge_index[1] == node_idx)[0].tolist()

                adjs = [walk_step['module'][0].edge_weight for walk_step in walk_steps]
                reverse_adjs = adjs.reverse()
                walk_indices = []
                walk_scores = []

                compute_walk_score(adjs, f, allow_edges)
                walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

            walks = {'ids': torch.tensor(walk_indices, device=self.device),
                     'score': torch.cat(walk_scores_tensor_list, dim=1)}

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
                edge_masks = []
                hard_edge_masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    edge_mask = edge_attr.detach()
                    valid_mask = (edge_mask != - math.inf)
                    edge_mask[edge_mask == - math.inf] = edge_mask[valid_mask].min() - 1  # replace the negative inf
                    edge_masks.append(edge_mask)
                    hard_edge_masks.append(self.control_sparsity(edge_attr, kwargs.get('sparsity')).sigmoid())

                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        return walks, edge_masks, related_preds
