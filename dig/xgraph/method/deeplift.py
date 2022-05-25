import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils.loop import add_remaining_self_loops
from ..models.utils import subgraph
from ..models.ext.deeplift.layer_deep_lift import DeepLift
from .base_explainer import WalkBase

EPS = 1e-15


class DeepLIFT(WalkBase):
    r"""
    An implementation of DeepLIFT on graph in
    `Learning Important Features Through Propagating Activation Differences <https://arxiv.org/abs/1704.02685>`_.
    Args:
        model (torch.nn.Module): The target model prepared to explain.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)
    .. note:: For node classification model, the :attr:`explain_graph` flag is False.
        For an example, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.
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
            **kwargs (dict): :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification) :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
        :rtype: (None, list, list)
        .. note::
            (None, edge_masks, related_predictions):
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.
        """

        # --- run the model once ---
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

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

        # --- add shap calculation hook ---
        shap = DeepLift(self.model)
        self.model.apply(shap._register_hooks)

        inp_with_ref = torch.cat([x, torch.zeros(x.shape, device=self.device, dtype=torch.float)], dim=0).requires_grad_(True)
        edge_index_with_ref = torch.cat([edge_index, edge_index + x.shape[0]], dim=1)
        batch = torch.arange(2, dtype=torch.long, device=self.device).view(2, 1).repeat(1, x.shape[0]).reshape(-1)
        out = self.model(inp_with_ref, edge_index_with_ref, batch)

        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        if kwargs.get('edge_masks'):
            edge_masks = kwargs.pop('edge_masks')
            hard_edge_masks = [self.control_sparsity(mask, kwargs.get('sparsity')).sigmoid() for mask in edge_masks]

        else:
            edge_masks = []
            hard_edge_masks = []
            for ex_label in ex_labels:

                if self.explain_graph:
                    f = torch.unbind(out[:, ex_label])
                else:
                    f = torch.unbind(out[[node_idx, node_idx + x.shape[0]], ex_label])

                (m, ) = torch.autograd.grad(outputs=f, inputs=inp_with_ref, retain_graph=True)
                inp, inp_ref = torch.chunk(inp_with_ref, 2)
                attr_wo_relu = (torch.chunk(m, 2)[0] * (inp - inp_ref)).sum(1)

                mask = attr_wo_relu.squeeze()
                score_mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2
                edge_masks.append(score_mask.detach())
                mask = self.control_sparsity(score_mask, kwargs.get('sparsity'))
                mask = mask.sigmoid()
                hard_edge_masks.append(mask.detach())

        # Store related predictions for further evaluation.
        shap._remove_hooks()

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        return edge_masks, hard_edge_masks, related_preds
