import math
import copy
import torch
import networkx as nx
from torch_geometric.data import Batch, Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
from Configures import mcts_args
from functools import partial
from collections import Counter
from shapley import gnn_score, NC_mc_l_shapley
from pipeline import MCTSNode

def k_hop_subgraph_with_default_whole_graph(node_idx, num_hops,
    edge_index, relabel_nodes=False, num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`. when num_hops == -1,
            the whole graph will be returned.
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

    if int(node_idx) == -1:
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


class MCTS():
    def __init__(self, node_idx: int, X: torch.Tensor, edge_index: torch.Tensor,
                 ori_graph: nx.Graph, n_rollout: int, min_atoms: int, c_puct: float,
                 expand_atoms: int, score_func=None, num_hops: int = 3):
        self.X = X
        self.edge_index = edge_index
        self.num_hops = num_hops
        self.ori_graph = ori_graph
        self.ori_node_idx = node_idx
        self.ori_num_nodes = self.ori_graph.number_of_nodes()

        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.score_func = score_func

        # extract the sub-graph and change the node indices.
        x, edge_index, subset, edge_mask, kwargs = \
            self.__subgraph__(node_idx, self.X, self.edge_index)
        self.data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        self.graph = self.ori_graph.subgraph(subset.tolist())
        mapping = {int(v): k for k, v in enumerate(subset)}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.node_idx = torch.where(subset == self.ori_node_idx)[0]
        self.num_nodes = self.graph.number_of_nodes()

        self.root_coalition = [i for i in range(self.num_nodes)]
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct)

        self.root = self.MCTSNodeClass(self.root_coalition)
        self.state_map = {str(sorted(self.root.coalition)): self.root}

    def set_score_func(self, score_func):
        self.score_func = score_func

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, subset, edge_mask, kwargs

    def mcts_rollout(self, tree_node):
        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=mcts_args.high2low)
            all_nodes = [x[0] for x in node_degree_list]
            all_degrees = [x[1] for x in node_degree_list]

            if len(all_nodes) < self.expand_atoms:
                expand_nodes = copy.copy(all_nodes)
            else:
                expand_nodes = copy.copy(all_nodes[:self.expand_atoms])
                expand_degrees = copy.copy(all_degrees[:self.expand_atoms])

            # keep the node_idx in the subgraph
            if self.node_idx in expand_nodes:
                expand_nodes.remove(self.node_idx)

            for each_node in expand_nodes:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and only keep the largest one
                subgraph_coalition = [node for node in all_nodes if node != each_node]

                subgraphs = [self.graph.subgraph(c)
                             for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]
                main_sub = subgraphs[0]
                for sub in subgraphs:
                    if sub.number_of_nodes() > main_sub.number_of_nodes():
                        main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))

                # check the state map and merge the same sub-graph
                Find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        Find_same = True

                if Find_same == False:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                Find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        Find_same_child = True

                if Find_same_child == False:
                    tree_node.children.append(new_node)

            scores = compute_scores(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score
        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))

        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"At the {rollout_idx} rollout, {len(self.state_map)} states that have been explored.")

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results


def reward_func(reward_args, value_func, node_idx=-1):
    if reward_args.reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method)

    elif reward_args.reward_method.lower() == 'nc_mc_l_shapley':
        return partial(NC_mc_l_shapley,
                       node_idx=node_idx,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method,
                       sample_num=reward_args.sample_num)

    else:
        raise NotImplementedError
