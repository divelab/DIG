import numpy as np
import torch
import torch.nn as nn
from .rgcn import *
from .st_net import *



class MaskedGraphAF(nn.Module):
    def __init__(self, mask_node, mask_edge, index_select_edge, st_type='sigmoid', num_flow_layer=12, graph_size=38,
                 num_node_type=9, num_edge_type=4, use_bn=True, num_rgcn_layer=3, nhid=128, nout=128):
        '''
        :param index_nod_edg:
        :param num_edge_type, virtual type included
        '''
        super(MaskedGraphAF, self).__init__()
        self.repeat_num = mask_node.size(0)
        self.graph_size = graph_size
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type

        self.mask_node = nn.Parameter(mask_node.view(1, self.repeat_num, graph_size, 1), requires_grad=False)  # (1, repeat_num, n, 1)
        self.mask_edge = nn.Parameter(mask_edge.view(1, self.repeat_num, 1, graph_size, graph_size), requires_grad=False)  # (1, repeat_num, 1, n, n)
        self.index_select_edge = nn.Parameter(index_select_edge, requires_grad=False)  # (edge_step_length, 2)

        self.emb_size = nout
        self.num_flow_layer = num_flow_layer

        self.rgcn = RGCN(num_node_type, nhid=nhid, nout=nout, edge_dim=self.num_edge_type-1,
                         num_layers=num_rgcn_layer, dropout=0., normalization=False)

        if use_bn:
            self.batchNorm = nn.BatchNorm1d(nout)

        self.st_type = st_type
        self.st_net_fn_dict = {'sigmoid': ST_Net_Sigmoid, 'exp': ST_Net_Exp, 'softplus': ST_Net_Softplus}
        assert st_type in ['sigmoid', 'exp', 'softplus'], 'unsupported st_type, choices are [sigmoid, exp, softplus, ]'
        st_net_fn = self.st_net_fn_dict[st_type]
        self.node_st_net = nn.ModuleList([st_net_fn(nout, self.num_node_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])
        self.edge_st_net = nn.ModuleList([st_net_fn(nout*3, self.num_edge_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])


    def forward(self, x, adj, x_deq, adj_deq):
        '''
        :param x:   (batch, N, 9)
        :param adj: (batch, 4, N, N)

        :param x_deq: (batch, N, 9)
        :param adj_deq:  (batch, edge_num, 4)
        :return:
        '''
        # inputs for RelGCNs
        batch_size = x.size(0)
        graph_emb_node, graph_node_emb_edge = self._get_embs(x, adj)
        x_deq = x_deq.view(-1, self.num_node_type)  # (batch *N, 9)
        adj_deq = adj_deq.view(-1, self.num_edge_type) # (batch*(repeat_num-N), 4)

        for i in range(self.num_flow_layer):
            # update x_deq
            node_s, node_t = self.node_st_net[i](graph_emb_node)
            if self.st_type == 'sigmoid':
                x_deq = x_deq * node_s + node_t
            elif self.st_type == 'exp':
                node_s = node_s.exp()
                x_deq = (x_deq + node_t) * node_s
            elif self.st_type == 'softplus':
                x_deq = (x_deq + node_t) * node_s
            else:
                raise ValueError('unsupported st type!')

            if torch.isnan(x_deq).any():
                raise RuntimeError('x_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()

            # update adj_deq
            edge_s, edge_t = self.edge_st_net[i](graph_node_emb_edge)
            if self.st_type == 'sigmoid':
                adj_deq = adj_deq * edge_s + edge_t
            elif self.st_type == 'exp':
                edge_s = edge_s.exp()
                adj_deq = (adj_deq + edge_t) * edge_s
            elif self.st_type == 'softplus':
                adj_deq = (adj_deq + edge_t) * edge_s
            else:
                raise ValueError('unsupported st type!')

            if torch.isnan(adj_deq).any():
                raise RuntimeError('adj_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                adj_log_jacob = (torch.abs(edge_s) + 1e-20).log()
            else:
                adj_log_jacob += (torch.abs(edge_s) + 1e-20).log()

        x_deq = x_deq.view(batch_size, -1)  # (batch, N * 9)
        adj_deq = adj_deq.view(batch_size, -1)  # (batch, (repeat_num-N) * 4)
        x_log_jacob = x_log_jacob.view(batch_size, -1).sum(-1)  # (batch)
        adj_log_jacob = adj_log_jacob.view(batch_size, -1).sum(-1)  # (batch)

        return [x_deq, adj_deq], [x_log_jacob, adj_log_jacob]        


    def forward_rl_node(self, x, adj, x_cont):
        """
        Args:
            x: shape (batch, N, 9)
            adj: shape (batch, 4, N, N)
            x_cont: shape (batch, 9)
        Returns:
            x_cont: shape (batch, 9)
            x_log_jacob: shape (batch, )
        """
        embs = self._get_embs_node(x, adj) # (batch, d)
        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](embs)

            if self.st_type == 'sigmoid':
                x_cont = x_cont * node_s + node_t
            elif self.st_type == 'exp':
                node_s = node_s.exp()
                x_cont = (x_cont + node_t) * node_s
            elif self.st_type == 'softplus':
                x_cont = (x_cont + node_t) * node_s
            else:
                raise ValueError('unsupported st type: (%s)' % self.args.st_type)

            if torch.isnan(x_cont).any():
                raise RuntimeError(
                    'x_cont has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()            

        x_log_jacob = x_log_jacob.sum(-1)  # (batch)

        return x_cont, x_log_jacob


    def forward_rl_edge(self, x, adj, x_cont, index):
        """
        Args:
            x: shape (batch, N, 9)
            adj: shape (batch, 4, N, N)
            x_cont: shape (batch, 4)
            index: shape (batch, 2)
        Returns:
            x_cont: shape (batch, 4)
            x_log_jacob: shape (batch, )            
        """
        embs = self._get_embs_edge(x, adj, index) # (batch, 3d)

        for i in range(self.num_flow_layer):
            edge_s, edge_t = self.edge_st_net[i](embs)

            if self.st_type == 'sigmoid':
                x_cont = x_cont * edge_s + edge_t
            elif self.st_type == 'exp':
                edge_s = edge_s.exp()
                x_cont = (x_cont + edge_t) * edge_s
            elif self.st_type == 'softplus':
                x_cont = (x_cont + edge_t) * edge_s
            else:
                raise ValueError('unsupported st type: (%s)' % self.args.st_type)

            if torch.isnan(x_cont).any():
                raise RuntimeError(
                    'x_cont has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(edge_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(edge_s) + 1e-20).log()            

        x_log_jacob = x_log_jacob.sum(-1)  # (batch)
        
        return x_cont, x_log_jacob        


    def reverse(self, x, adj, latent, mode, edge_index=None):
        '''
        Args:
            x: generated subgraph node features so far with shape (1, N, 9), some part of the x is masked
            adj: generated subgraph adacency features so far with shape (1, 4, N, N) some part of the adj is masked
            latent: sample latent vector with shape (1, 9) (mode == 0) or (1, 4) (mode == 1)
            mode: generation mode. if mode == 0, generate a new node, if mode == 1, generate a new edge
            edge_index [1, 2]

        Returns:
            out: generated node/edge features with shape (1, 9) (mode == 0) or (1, 4) , (mode == 1)
        '''

        assert mode == 0 or edge_index is not None, 'if you want to generate edge, you must specify edge_index'
        assert x.size(0) == 1
        assert adj.size(0) == 1
        assert edge_index is None or (edge_index.size(0) == 1 and edge_index.size(1) == 2)
        
        if mode == 0: #(1, 9)
            st_net = self.node_st_net
            emb = self._get_embs_node(x, adj)

        else:  # mode == 1
            st_net = self.edge_st_net
            emb = self._get_embs_edge(x, adj, edge_index)            

        for i in reversed(range(self.num_flow_layer)):
            s, t = st_net[i](emb)
            if self.st_type == 'sigmoid':
                latent = (latent - t) / s
            elif self.st_type == 'exp':
                s = s.exp()
                latent = (latent / s) - t
            elif self.st_type == 'softplus':
                latent = (latent / s) - t
            else:
                raise ValueError('unsupported st type')

        return latent


    def _get_embs_node(self, x, adj):
        """
        Args:
            x: current node feature matrix with shape (batch, N, 9)
            adj: current adjacency feature matrix with shape (batch, 4, N, N)
        Returns:
            graph embedding for updating node features with shape (batch, d)
        """
        adj = adj[:, :3] # (batch, 3, N, N)

        node_emb = self.rgcn(x, adj) # (batch, N, d)
        if hasattr(self, 'batchNorm'):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2) # (batch, N, d)
        
        graph_emb = torch.sum(node_emb, dim=1, keepdim=False).contiguous() # (batch, d)
        return graph_emb


    def _get_embs_edge(self, x, adj, index):
        """
        Args:
            x: current node feature matrix with shape (batch, N, 9)
            adj: current adjacency feature matrix with shape (batch, 4, N, N)
            index: link prediction index with shape (batch, 2)
        Returns:
            Embedding(concatenate graph embedding, edge start node embedding and edge end node embedding) 
                for updating edge features with shape (batch, 3d)
        """
        batch_size = x.size(0)
        assert batch_size == index.size(0)

        adj = adj[:, :3] # (batch, 3, N, N)

        node_emb = self.rgcn(x, adj) # (batch, N, d)
        if hasattr(self, 'batchNorm'):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2) # (batch, N, d)

        graph_emb = torch.sum(node_emb, dim = 1, keepdim=False).contiguous().view(batch_size, 1, -1) # (batch, 1, d)

        index = index.view(batch_size, -1, 1).repeat(1, 1, self.emb_size) # (batch, 2, d)
        graph_node_emb = torch.cat((torch.gather(node_emb, dim=1, index=index), 
                                        graph_emb),dim=1)  # (batch_size, 3, d)
        graph_node_emb = graph_node_emb.view(batch_size, -1) # (batch_size, 3d)
        return graph_node_emb


    def _get_embs(self, x, adj):
        '''
        :param x of shape (batch, N, 9)
        :param adj of shape (batch, 4, N, N)
        :return: inputs for st_net_node and st_net_edge
        graph_emb_node of shape (batch*N, d)
        graph_emb_edge of shape (batch*(repeat-N), 3d)

        '''
        # inputs for RelGCNs
        batch_size = x.size(0)
        adj = adj[:, :3] # (batch, 3, N, N) TODO: check whether we have to use the 4-th slices(virtual bond) or not
        x = torch.where(self.mask_node, x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1), torch.zeros([1], device=x.device)).view(
            -1, self.graph_size, self.num_node_type)  # (batch*repeat_num, N, 9)

        adj = torch.where(self.mask_edge, adj.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1), torch.zeros([1], device=x.device)).view(
            -1, self.num_edge_type - 1, self.graph_size, self.graph_size)  # (batch*repeat_num, 3, N, N)
        node_emb = self.rgcn(x, adj)  # (batch*repeat_num, N, d)

        if hasattr(self, 'batchNorm'):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch*repeat_num, N, d)

        node_emb = node_emb.view(batch_size, self.repeat_num, self.graph_size, -1) # (batch, repeat_num, N, d)


        graph_emb = torch.sum(node_emb, dim=2, keepdim=False) # (batch, repeat_num, d)

        #  input for st_net_node
        graph_emb_node = graph_emb[:, :self.graph_size].contiguous() # (batch, N, d)
        graph_emb_node = graph_emb_node.view(batch_size * self.graph_size, -1)  # (batch*N, d)

        # input for st_net_edge
        graph_emb_edge = graph_emb[:, self.graph_size:].contiguous() # (batch, repeat_num-N, d)
        graph_emb_edge = graph_emb_edge.unsqueeze(2)  # (batch, repeat_num-N, 1, d)

        all_node_emb_edge = node_emb[:, self.graph_size:] # (batch, repeat_num-N, N, d)

        index = self.index_select_edge.view(1, -1, 2, 1).repeat(batch_size, 1, 1,
                                        self.emb_size)  # (batch_size, repeat_num-N, 2, d)


        graph_node_emb_edge = torch.cat((torch.gather(all_node_emb_edge, dim=2, index=index), 
                                        graph_emb_edge),dim=2)  # (batch_size, repeat_num-N, 3, d)

        graph_node_emb_edge = graph_node_emb_edge.view(batch_size * (self.repeat_num - self.graph_size),
                                        -1)  # (batch_size * (repeat_num-N), 3*d)

        return graph_emb_node, graph_node_emb_edge