import torch
from torch import Tensor
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.version import debug
from ..models.utils import subgraph
from torch.nn.functional import cross_entropy
from .base_explainer import ExplainerBase
from typing import Union
EPS = 1e-15


def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)


class GNNExplainer(ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.
    .. note:: For an example, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        explain_graph (bool, optional): Whether to explain graph classification model
            (default: :obj:`False`)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int = 100,
                 lr: float = 0.01,
                 coff_size: float = 0.001,
                 coff_ent: float = 0.001,
                 explain_graph: bool = False):
        super(GNNExplainer, self).__init__(model, epochs, lr, explain_graph)
        self.coff_ent = coff_ent
        self.coff_size = coff_size

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]):
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            loss = cross_entropy_with_logit(raw_preds[self.node_idx].reshape(1, -1), x_label)

        m = self.edge_mask.sigmoid()
        loss = loss + self.coff_size * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coff_ent * ent.mean()

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
                          ) -> Tensor:

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
            if epoch % 20 == 0 and debug:
                print(f'Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
            optimizer.step()

        return self.edge_mask.data

    def forward(self, x, edge_index, mask_features=False, **kwargs):
        r"""
        Run the explainer for a specific graph instance.
        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            mask_features (bool, optional): Whether to use feature mask. Not recommended.
                (Default: :obj:`False`)
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.
        :rtype: (None, list, list)
        .. note::
            (None, edge_masks, related_predictions):
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.
        """
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            if not node_idx.dim():
                node_idx = node_idx.reshape(-1)
            node_idx = node_idx.to(self.device)
            assert node_idx is not None
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.node_idx = node_idx
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

        if kwargs.get('edge_masks'):
            edge_masks = kwargs.pop('edge_masks')
            self.__set_masks__(x, self_loop_edge_index)

        else:
            # Assume the mask we will predict
            labels = tuple(i for i in range(kwargs.get('num_classes')))
            ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

            # Calculate mask
            edge_masks = []
            for ex_label in ex_labels:
                self.__clear_masks__()
                self.__set_masks__(x, self_loop_edge_index)
                edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))

        hard_edge_masks = [self.control_sparsity(mask, sparsity=kwargs.get('sparsity')).sigmoid()
                           for mask in edge_masks]

        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds

    def __repr__(self):
        return f'{self.__class__.__name__}()'
