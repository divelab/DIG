import torch
import torch.nn as nn
from torch_geometric.utils.loop import add_self_loops
from dig.xgraph.models.utils import subgraph
from dig.xgraph.method.base_explainer import ExplainerBase


class RandomSelectorExplainer(ExplainerBase):
    def __init__(self, model: nn.Module, explain_graph: bool = False):
        super().__init__(model=model, explain_graph=explain_graph)

    def forward(self, x, edge_index, **kwargs):
        super().forward(x, edge_index)
        self.model.eval()

        # Assume the mask we will predict
        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        if self.explain_graph:
            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
            edge_mask = torch.rand(self_loop_edge_index.shape[1])
            edge_masks = [edge_mask for _ in ex_labels]

            self.__clear_masks__()
            self.__set_masks__(x, self_loop_edge_index)
            hard_edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity')).sigmoid().to(self.device)
                               for _ in ex_labels]

            with torch.no_grad():
                related_preds = self.eval_related_pred(
                    x, edge_index, hard_edge_masks)
            self.__clear_masks__()

        else:
            node_idx = kwargs.get('node_idx')
            if not node_idx.dim():
                node_idx = node_idx.reshape(-1)
            node_idx = node_idx.to(self.device)
            assert node_idx is not None

            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index,
                relabel_nodes=True, num_nodes=None, flow=self.__flow__())

            edge_mask = torch.rand(self_loop_edge_index.shape[1])

            self.__clear_masks__()
            self.__set_masks__(x, self_loop_edge_index)
            edge_masks = [edge_mask for _ in ex_labels]
            hard_edge_masks = [self.control_sparsity(
                edge_mask, sparsity=kwargs.get('sparsity')).sigmoid().to(self.device) for _ in ex_labels]

            with torch.no_grad():
                related_preds = self.eval_related_pred(
                    x, edge_index, hard_edge_masks, **kwargs)
            self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds
