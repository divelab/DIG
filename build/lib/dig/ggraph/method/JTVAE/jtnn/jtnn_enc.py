import torch
import torch.nn as nn
from collections import deque
from .mol_tree import Vocab, MolTree
from .nnutils import create_var, GRU

MAX_NB = 8

class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, root_batch):
        orders = []
        for root in root_batch:
            order = get_prop_order(root)
            orders.append(order)
        
        h = {}
        max_depth = max([len(x) for x in orders])
        padding = create_var(torch.zeros(self.hidden_size), False)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x,node_y in prop_list:
                x,y = node_x.idx,node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[(z,x)])

                pad_len = MAX_NB - len(h_nei)
                h_nei.extend([padding] * pad_len)
                cur_h_nei.extend(h_nei)

            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)

            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i,m in enumerate(prop_list):
                x,y = m[0].idx,m[1].idx
                h[(x,y)] = new_h[i]

        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)

        return h, root_vecs

"""
Helper functions
"""

def get_prop_order(root):
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1,order2 = [],[]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x,y) )
                order2[y.depth-1].append( (y,x) )
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = create_var(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))

