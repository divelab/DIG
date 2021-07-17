import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab, MolTree
from .nnutils import create_var, index_select_ND


class JTNNEncoder(nn.Module):

    def __init__(self, hidden_size, depth, embedding):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.embedding = embedding
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope):
        fnode = create_var(fnode)
        fmess = create_var(fmess)
        node_graph = create_var(node_graph)
        mess_graph = create_var(mess_graph)
        messages = create_var(torch.zeros(
            mess_graph.size(0), self.hidden_size))

        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, 0, fmess)
        messages = self.GRU(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        node_vecs = self.outputNN(node_vecs)

        max_len = max([x for _, x in scope])
        batch_vecs = []
        for st, le in scope:
            cur_vecs = node_vecs[st]  # Root is the first node
            batch_vecs.append(cur_vecs)

        tree_vecs = torch.stack(batch_vecs, dim=0)
        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):
        node_batch = []
        scope = []
        for tree in tree_batch:
            scope.append((len(node_batch), len(tree.nodes)))
            node_batch.extend(tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope)

    @staticmethod
    def tensorize_nodes(node_batch, scope):
        messages, mess_dict = [None], {}
        fnode = []
        for x in node_batch:
            fnode.append(x.wid)
            for y in x.neighbors:
                mess_dict[(x.idx, y.idx)] = len(messages)
                messages.append((x, y))

        node_graph = [[] for i in range(len(node_batch))]
        mess_graph = [[] for i in range(len(messages))]
        fmess = [0] * len(messages)

        for x, y in messages[1:]:
            mid1 = mess_dict[(x.idx, y.idx)]
            fmess[mid1] = x.idx
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx:
                    continue
                mid2 = mess_dict[(y.idx, z.idx)]
                mess_graph[mid2].append(mid1)

        max_len = max([len(t) for t in node_graph] + [1])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        mess_graph = torch.LongTensor(mess_graph)
        node_graph = torch.LongTensor(node_graph)
        fmess = torch.LongTensor(fmess)
        fnode = torch.LongTensor(fnode)
        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict


class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0  # first vector is padding
        mask = create_var(mask)
        for it in range(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = h_nei.sum(dim=1)
            z_input = torch.cat([x, sum_h], dim=1)
            z = F.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = F.sigmoid(r_1 + r_2)

            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = torch.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h
