"""
FileName: models.py
Description: GNN models' set
Time: 2020/7/30 9:01
Project: GNN_benchmark
Author: Shurui Gui
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as gnn
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from benchmark.args import data_args
from torch_geometric.data.batch import Batch

from typing import Callable, Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor

from torch_sparse import SparseTensor


class GNNBasic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def arguments_read(self, *args, **kwargs):

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        return x, edge_index, batch

class GCN_3l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 3

        self.conv1 = GCNConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out


class GCN_2l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 2

        self.conv1 = GCNConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)

        return out


class GIN_3l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 3

        self.conv1 = GINConv(nn.Sequential(nn.Linear(data_args.dim_node, data_args.dim_hidden), nn.ReLU(),
                                           nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(data_args.dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU(),
                                      nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(data_args.dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

class GIN_2l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 2

        self.conv1 = GINConv(nn.Sequential(nn.Linear(data_args.dim_node, data_args.dim_hidden), nn.ReLU(),
                                           nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(data_args.dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU(),
                                      nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(data_args.dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out


class GCNConv(gnn.GCNConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out


class GINConv(gnn.GINConv):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.edge_weight = None
        self.fc_steps = None
        self.reweight = None

    # def children(self):
    #     if
    #     return iter([])


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, **kwargs) -> Tensor:
        """"""
        self.num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if edge_weight is not None:
            self.edge_weight = edge_weight
            assert edge_weight.shape[0] == edge_index.shape[1]
            self.reweight = False
        else:
            edge_index, _ = remove_self_loops(edge_index)
            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            if self_loop_edge_index.shape[1] != edge_index.shape[1]:
                edge_index = self_loop_edge_index
            self.reweight = True
        out = self.propagate(edge_index, x=x[0], size=None)

        if data_args.task == 'explain':
            layer_extractor = []
            hooks = []

            def register_hook(module: nn.Module):
                if not list(module.children()):
                    hooks.append(module.register_forward_hook(forward_hook))

            def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
                # input contains x and edge_index
                layer_extractor.append((module, input[0], output))

            # --- register hooks ---
            self.nn.apply(register_hook)

            nn_out = self.nn(out)

            for hook in hooks:
                hook.remove()

            fc_steps = []
            step = {'input': None, 'module': [], 'output': None}
            for layer in layer_extractor:
                if isinstance(layer[0], nn.Linear):
                    if step['module']:
                        fc_steps.append(step)
                    # step = {'input': layer[1], 'module': [], 'output': None}
                    step = {'input': None, 'module': [], 'output': None}
                step['module'].append(layer[0])
                if kwargs.get('probe'):
                    step['output'] = layer[2]
                else:
                    step['output'] = None

            if step['module']:
                fc_steps.append(step)
            self.fc_steps = fc_steps
        else:
            nn_out = self.nn(out)


        return nn_out

    def message(self, x_j: Tensor) -> Tensor:
        if self.reweight:
            edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
            edge_weight.data[-self.num_nodes:] += self.eps
            edge_weight = edge_weight.detach().clone()
            edge_weight.requires_grad_(True)
            self.edge_weight = edge_weight
        return x_j * self.edge_weight.view(-1, 1)


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x


class GraphSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *input) -> Tensor:
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

