import torch
from torch import nn
from .st_net import ST_Dis
from .df_utils import one_hot_add, one_hot_minus
from .rgcn import RGCN

class DisGraphAF(nn.Module):
    def __init__(self, mask_node, mask_edge, index_select_edge, num_flow_layer=12, graph_size=38,
                 num_node_type=9, num_edge_type=4, use_bn=True, num_rgcn_layer=3, nhid=128, nout=128):
        '''
        :param index_nod_edg:
        :param num_edge_type, virtual type included
        '''
        super(DisGraphAF, self).__init__()
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

        self.node_st_net = nn.ModuleList([ST_Dis(nout, self.num_node_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])
        self.edge_st_net = nn.ModuleList([ST_Dis(nout*3, self.num_edge_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])
        

    def forward(self, x, adj, x_deq, adj_deq):
        '''
        :param x:   (batch, N, 9)
        :param adj: (batch, 4, N, N)
        
        :param x_deq: (batch, N, 9)
        :param adj_deq:  (batch, edge_num, 4)
        :return:
        x_deq: (batch, N, 9)
        adj_deq: (batch, edge_num, 4)
        '''
        # inputs for RelGCNs
        graph_emb_node, graph_node_emb_edge = self._get_embs(x, adj)

        for i in range(self.num_flow_layer):
            # update x_deq
            node_t = self.node_st_net[i](graph_emb_node).type(x.dtype)
            x_deq = one_hot_add(x_deq, node_t)

            # update adj_deq
            edge_t = self.edge_st_net[i](graph_node_emb_edge).type(adj.dtype)
            adj_deq = one_hot_add(adj_deq, edge_t)

        return [x_deq, adj_deq]


    def forward_rl_node(self, x, adj, x_cont):
        """
        Args:
            x: shape (batch, N, 9)
            adj: shape (batch, 4, N, N)
            x_cont: shape (batch, 9)
        Returns:
            x_cont: shape (batch, 9)
        """
        embs = self._get_embs_node(x, adj) # (batch, d)
        for i in range(self.num_flow_layer):
            node_t = self.node_st_net[i](embs)
            x_cont = one_hot_add(x_cont, node_t)

        return x_cont, None


    def forward_rl_edge(self, x, adj, x_cont, index):
        """
        Args:
            x: shape (batch, N, 9)
            adj: shape (batch, 4, N, N)
            x_cont: shape (batch, 4)
            index: shape (batch, 2)
        Returns:
            x_cont: shape (batch, 4)            
        """
        embs = self._get_embs_edge(x, adj, index) # (batch, 3d)
        for i in range(self.num_flow_layer):
            edge_t = self.edge_st_net[i](embs)
            x_cont = one_hot_add(x_cont, edge_t)
        
        return x_cont, None


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
            t = st_net[i](emb)
            latent = one_hot_minus(latent, t)

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
        graph_emb_node of shape (batch, N, d)
        graph_emb_edge of shape (batch, repeat-N, 3d)

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
        # graph_emb_node = graph_emb_node.view(batch_size * self.graph_size, -1)  # (batch*N, d)

        # input for st_net_edge
        graph_emb_edge = graph_emb[:, self.graph_size:].contiguous() # (batch, repeat_num-N, d)
        graph_emb_edge = graph_emb_edge.unsqueeze(2)  # (batch, repeat_num-N, 1, d)

        all_node_emb_edge = node_emb[:, self.graph_size:] # (batch, repeat_num-N, N, d)

        index = self.index_select_edge.view(1, -1, 2, 1).repeat(batch_size, 1, 1,
                                        self.emb_size)  # (batch_size, repeat_num-N, 2, d)


        graph_node_emb_edge = torch.cat((torch.gather(all_node_emb_edge, dim=2, index=index), 
                                        graph_emb_edge),dim=2)  # (batch_size, repeat_num-N, 3, d)

        graph_node_emb_edge = graph_node_emb_edge.view(batch_size, self.repeat_num - self.graph_size,
                                        -1)  # (batch_size, (repeat_num-N), 3*d)

        return graph_emb_node, graph_node_emb_edge