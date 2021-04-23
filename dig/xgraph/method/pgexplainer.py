"""
Description: The implement of PGExplainer model
<https://arxiv.org/abs/2011.04573>
"""

from typing import Optional
from math import sqrt

import time
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data, Batch
import tqdm
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple, List, Dict
from .shapley import GnnNets_GC2value_func, GnnNets_NC2value_func, gnn_score


EPS = 1e-6


def inv_sigmoid(t: torch.Tensor):
    """ except the case t is 0 or 1 """
    if t.shape[0] != 0:
        if t.min().item() == 0 or t.max().item() == 1:
            t = 0.99 * t + 0.005
    ret = - torch.log(1 / t - 1)
    return ret


def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=3, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx


def calculate_selected_nodes(data, edge_mask, top_k):
    threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
    hard_mask = (edge_mask > threshold).cpu()
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    edge_index = data.edge_index.cpu().numpy()
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
    selected_nodes = list(set(selected_nodes))
    return selected_nodes


class PGExplainer(nn.Module):
    r"""
    An implementation of `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_

    Args:
        model (torch.nn.Module):
        in_channels (int):
        explain_graph (bool): Whether to explain graph classification model
        epochs (int): Number of epochs for training
        lr (float): learning rate
        coff_size (float):
        coff_ent (float):
        t0 (float):
        t1(float):
        num_hops (int, optional):

    .. notes: For graph classification model, the explain_graph flag should be True, and it should be false
                when the model is node classification task.

    """
    def __init__(self, model, in_channels: int, explain_graph: bool = True, epochs: int = 20,
                 lr: float = 0.003, coff_size: float = 1.0, coff_ent: float = 1.0,
                 t0: float = 1.0, t1: float = 1.0, num_hops: Optional[int] = None):
        super(PGExplainer, self).__init__()
        self.model = model
        self.in_channels = in_channels
        self.explain_graph = explain_graph

        # training parameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1

        self.num_hops = self.update_num_hops(num_hops)
        self.device = model.device
        self.init_bias = 0.0

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask:Tensor =None):
        """ Set the edge weights before message passing """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    def update_num_hops(self, num_hops):
        """ Return the number of layers of GNN model """
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __loss__(self, prob, ori_pred):
        """
        the pred loss encourages the masked graph with higher probability,
        the size loss encourage small size edge mask,
        the entropy loss encourage the mask to be continuous.
        """
        logit = prob[ori_pred]
        logit = logit + EPS
        pred_loss = - torch.log(logit)
        # size
        edge_mask = torch.sigmoid(self.mask_sigmoid)
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def get_model_output(self, x, edge_index, edge_mask=None):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()
        self.__clear_masks__()
        if edge_mask is not None:
            self.__set_masks__(x, edge_index, edge_mask.to(self.device))

        with torch.no_grad():
            data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
            data.to(self.device)
            outputs = self.model(data)

        self.__clear_masks__()
        return outputs

    def get_subgraph(self, node_idx, x, edge_index, y=None, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, self.num_hops, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        if y is not None:
            y = y[subset]
        return x, edge_index, y, subset, kwargs

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """
        Sample from the instantiation of concrete distribution when training
        """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def explain(self, x: Tensor, edge_index: Tensor, embed: Tensor,
                tmp: float = 1.0, training: bool = False):

        nodesize = embed.shape[0]
        feature_dim = embed.shape[1]
        f1 = embed.unsqueeze(1).repeat(1, nodesize, 1).reshape(-1, feature_dim)
        f2 = embed.unsqueeze(0).repeat(nodesize, 1, 1).reshape(-1, feature_dim)

        # using the node embedding to calculate the edge weight
        f12self = torch.cat([f1, f2], dim=-1)
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.mask_sigmoid = values.reshape(nodesize, nodesize)

        # set the symmetric edge weights
        sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        # inverse the weights before sigmoid in MessagePassing Module
        edge_mask = inv_sigmoid(edge_mask)
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        data.to(self.device)
        outputs = self.model(data)
        return outputs[1].squeeze(), edge_mask

    def train_explanation_network(self, dataset):
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        if self.explain_graph:
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    _, prob, emb = self.get_model_output(data.x, data.edge_index)
                    emb_dict[gid] = emb.data.cpu()
                    ori_pred_dict[gid] = prob.argmax(-1).data.cpu()

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                pred_list = []
                acc_list = []
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    prob, _ = self.explain(data.x, data.edge_index, embed=emb_dict[gid], tmp=tmp, training=True)
                    loss_tmp = self.__loss__(prob, ori_pred_dict[gid])
                    loss_tmp.backward()
                    loss += loss_tmp.item()
                    pred_label = prob.argmax(-1).item()
                    pred_list.append(pred_label)
                    acc_list.append(pred_label == data.y)

                optimizer.step()
                duration += time.perf_counter() - tic
                accs = torch.stack(acc_list, dim=0)
                acc = np.array(accs).mean()
                print(f'Epoch: {epoch} | Loss: {loss} | Acc : {acc}')
        else:
            with torch.no_grad():
                data = dataset[0]
                self.model.eval()
                x_dict = {}
                edge_index_dict = {}
                node_idx_dict = {}
                pred_dict = {}
                emb_dict = {}
                for node_idx in tqdm.tqdm(range(data.x.shape[0])):
                    x, edge_index, y, subset, _ = \
                        self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
                    _, prob, emb = self.get_model_output(data.x, data.edge_index)
                    x_dict[node_idx] = x
                    edge_index_dict[node_idx] = edge_index
                    node_idx_dict[node_idx] = int(torch.where(subset == node_idx)[0])
                    pred_dict[node_idx] = prob[node_idx_dict[node_idx]].argmax(-1).cpu()
                    emb_dict[node_idx] = emb.data.cpu()

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                acc_list = []
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                tic = time.perf_counter()
                for node_idx in tqdm.tqdm(range(data.x.shape[0])):
                    pred, edge_mask = self.explain(x_dict[node_idx], edge_index_dict[node_idx],
                                                   emb_dict[node_idx], tmp, training=True)
                    loss_tmp = self.__loss__(pred[node_idx_dict[node_idx]], pred_dict[node_idx])
                    loss_tmp.backward()
                    loss += loss_tmp.item()

                    acc_list.append(pred[node_idx_dict[node_idx]].argmax().item() == data.y[node_idx])

                optimizer.step()
                duration += time.perf_counter() - tic
                accs = torch.stack(acc_list, dim=0)
                acc = np.array(accs).mean()
                print(f'Epoch: {epoch} | Loss: {loss} | Acc : {acc}')
            print(f"training time is {duration:.5}s")

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs)\
            -> Tuple[None, List, List[Dict]]:
        # set default subgraph with 10 edges
        top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10
        y = kwargs.get('y')
        _, probs, embed = self.get_model_output(x, edge_index, edge_mask=None)

        if self.explain_graph:
            # original value
            probs = probs.squeeze()
            label = y
            # masked value
            _, edge_mask = self.explain(x, edge_index, embed=embed, tmp=1.0, training=False)
            data = Data(x=x, edge_index=edge_index)
            selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
            masked_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNets_GC2value_func(self.model, target_class=label)
            masked_pred = gnn_score(masked_nodes_list, data, value_func,
                                    subgraph_building_method='zero_filling')
            sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]
        else:
            node_idx = kwargs.get('node_idx')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            # original value
            probs = probs.squeeze()[node_idx]
            label = y[node_idx]
            # masked value
            x, edge_index, _, subset, _ = self.get_subgraph(node_idx, x, edge_index)
            new_node_idx = torch.where(subset == node_idx)[0]
            _, edge_mask = self.explain(x, edge_index, embed[node_idx], tmp=1.0, training=False)
            data = Data(x=x, edge_index=edge_index)
            selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
            masked_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNets_NC2value_func(self.model,
                                               node_idx=new_node_idx,
                                               target_class=label)
            masked_pred = gnn_score(masked_nodes_list, data, value_func,
                                    subgraph_building_method='zero_filling')
            sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]

        # return variables
        pred_mask = [edge_mask]
        related_preds = [{'masked': masked_pred,
                         'origin': probs[label],
                         'sparsity': sparsity_score}]
        return None, pred_mask, related_preds

    def __repr__(self):
        return f'{self.__class__.__name__}()'
