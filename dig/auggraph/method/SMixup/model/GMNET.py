import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GINConv

class Readout(nn.Module):
    def __init__(self, node_feat_dim, node_hiddens, graph_hiddens, use_gate=True, pool_type='add'):
        super(Readout, self).__init__()
        self.graph_feat_dim = node_hiddens[-1]
        self.use_gate = use_gate
        self._get_node_net(node_feat_dim, node_hiddens, use_gate)
        self._get_graph_net(self.graph_feat_dim, graph_hiddens)

        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'sum':
            self.pool = global_add_pool
        elif pool_type == 'max':
            self.pool = global_max_pool

    def _get_node_net(self, node_feat_dim, node_hiddens, use_gate):
        if use_gate:
            node_hiddens[-1] *= 2
        
        layer = []
        layer.append(nn.Linear(node_feat_dim, node_hiddens[0]))
        for i in range(1, len(node_hiddens)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hiddens[i - 1], node_hiddens[i]))
        
        self.node_net = nn.Sequential(*layer)
    
    def _get_graph_net(self, graph_feat_dim, graph_hiddens):
        layer = []
        layer.append(nn.Linear(graph_feat_dim, graph_hiddens[0]))
        for i in range(1, len(graph_hiddens)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(graph_hiddens[i - 1], graph_hiddens[i]))
        
        self.graph_net = nn.Sequential(*layer)
    
    def forward(self, x, batch):
        x = self.node_net(x)
        if self.use_gate:
            gates = torch.sigmoid(x[:, :self.graph_feat_dim])
            x = x[:, self.graph_feat_dim:] * gates
        graph_feat = self.pool(x, batch)
        graph_feat = self.graph_net(graph_feat)

        return graph_feat


class GMNConv(nn.Module):
    def __init__(self, node_feat_dim, message_net_hiddens, update_net_hiddens, node_update_type='residual', layer_norm=False):
        super(GMNConv, self).__init__()
        assert node_update_type in ['mlp', 'residual', 'gru']
        self.node_update_type = node_update_type
        self._get_message_net(node_feat_dim, message_net_hiddens)
        self._get_update_net(node_feat_dim, update_net_hiddens, node_update_type)
        if layer_norm:
            self.message_norm = nn.LayerNorm()
            self.update_norm = nn.LayerNorm()
    
    def _get_message_net(self, node_feat_dim, message_net_hiddens):
        layer = []
        layer.append(nn.Linear(node_feat_dim * 2, message_net_hiddens[0]))
        for i in range(1, len(message_net_hiddens)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(message_net_hiddens[i - 1], message_net_hiddens[i]))
        self.message_net = nn.Sequential(*layer)
    
    def _get_update_net(self, node_feat_dim, update_net_hiddens, node_update_type):
        if node_update_type == 'gru':
            self.update_net = nn.GRU(node_feat_dim * 3, node_feat_dim)
        else:
            layer = []
            layer.append(nn.Linear(node_feat_dim * 4, update_net_hiddens[0]))
            for i in range(1, len(update_net_hiddens)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(update_net_hiddens[i - 1], update_net_hiddens[i]))
            layer.append(nn.ReLU())
            layer.append(nn.Linear(update_net_hiddens[-1], node_feat_dim))
            self.update_net = nn.Sequential(*layer)
    
    def message_aggr(self, x, edge_index):
        target_idx, source_idx = edge_index
        message_inputs = torch.cat((x.index_select(0, source_idx), x.index_select(0, target_idx)), dim=-1)
        messages = self.message_net(message_inputs)
        aggregation = scatter(messages, target_idx, dim=0, dim_size=x.shape[0], reduce='add')

        if hasattr(self, 'message_norm'):
            aggregation = self.message_norm(aggregation)
        return aggregation
    
    def node_update(self, x, messages, attentions):
        if self.node_update_type == 'gru':
            update_inputs = torch.cat((messages, attentions), dim=-1)
            _, new_x = self.update_net(update_inputs.unsqueeze(0), x.unsqueeze(0))
            new_x = torch.squeeze(new_x)
        else:
            update_inputs = torch.cat((messages, attentions, x), dim=-1)
            new_x = self.update_net(update_inputs)

        if hasattr(self, 'update_norm'):
            new_x = self.update_norm(new_x)
        
        if self.node_update_type == 'residual':
            return x + new_x
        return new_x
    
    def cross_attention1(self, x1, batch1, x2, batch2):
        results1, results2 = [], []
        for i in range(batch1[-1] + 1):
            x, y = x1[batch1 == i, :], x2[batch2 == i, :]
            att_scores = torch.mm(x, torch.transpose(y, 1, 0))
            attention_x = torch.mm(torch.softmax(att_scores, dim=1), y)
            attention_y = torch.mm(torch.transpose(torch.softmax(att_scores, dim=0), 1, 0), x)
            results1.append(attention_x)
            results2.append(attention_y)
        
        attentions1 = x1 - torch.cat(results1, dim=0)
        attentions2 = x2 - torch.cat(results2, dim=0)

        return attentions1, attentions2
    
    def cross_attention2(self, x1, batch1, x2, batch2):
        att_scores = torch.mm(x1, x2.T)
        att_scores[batch1.view(-1,1) != batch2.view(1,-1)] = -float('inf')

        att_weights1 = F.softmax(att_scores, dim=1)
        att_weights2 = F.softmax(att_scores.T, dim=1)
        attentions1 = x1 - torch.mm(att_weights1, x2)
        attentions2 = x2 - torch.mm(att_weights2, x1)

        return attentions1, attentions2

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        messages1 = self.message_aggr(x1, edge_index1)
        messages2 = self.message_aggr(x2, edge_index2)
        
        attentions1, attentions2 = self.cross_attention2(x1, batch1, x2, batch2)
        
        new_x1 = self.node_update(x1, messages1, attentions1)
        new_x2 = self.node_update(x2, messages2, attentions2)

        return new_x1, new_x2


class GMNet(nn.Module):
    def __init__(self, in_dim, num_layers, hidden, pool_type='sum', use_gate=True, node_update_type='residual', layer_norm=False, ogb = False):
        super(GMNet, self).__init__()
        

        self.embedding = nn.Sequential(
                nn.Linear(in_dim, hidden),
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GMNConv(hidden, [hidden * 2, hidden * 2], [hidden * 2], node_update_type, layer_norm))
        
        self.readout = Readout(hidden, [hidden], [hidden], use_gate, pool_type)
    
    def embed(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        
        x1, x2 = self.embedding(x1), self.embedding(x2)
        for conv in self.convs:
            x1, x2 = conv(x1, edge_index1, batch1, x2, edge_index2, batch2)

        embed1, embed2 = self.readout(x1, batch1), self.readout(x2, batch2)
        return embed1, embed2

    def node_embed(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        
        x1, x2 = self.embedding(x1), self.embedding(x2)
        for conv in self.convs:
            x1, x2 = conv(x1, edge_index1, batch1, x2, edge_index2, batch2)

        return x1, x2
    
    def forward(self, data1, data2, node_emd = False):
        if node_emd:
            embed1, embed2 = self.node_embed(data1, data2)
        else:
            embed1, embed2 = self.embed(data1, data2)
        return embed1, embed2