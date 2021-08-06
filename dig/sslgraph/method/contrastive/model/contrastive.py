import os
import torch
from tqdm import trange
import torch.nn as nn
from torch_geometric.data import Batch, Data
from dig.sslgraph.method.contrastive.objectives import NCE_loss, JSE_loss


class Contrastive(nn.Module):
    r"""
    Base class for creating contrastive learning models for either graph-level or 
    node-level tasks.
    
    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`Contrastive`.

    Args:
        objective (string, or callable): The learning objective of contrastive model.
            If string, should be one of 'NCE' and 'JSE'. If callable, should take lists
            of representations as inputs and returns loss Tensor 
            (see `dig.sslgraph.method.contrastive.objectives` for examples).
        views_fn (list of callable): List of functions to generate views from given graphs.
        graph_level (bool, optional): Whether to include graph-level representation 
            for contrast. (default: :obj:`True`)
        node_level (bool, optional): Whether to include node-level representation 
            for contrast. (default: :obj:`False`)
        z_dim (int, optional): The dimension of graph-level representations. 
            Required if :obj:`graph_level` = :obj:`True`. (default: :obj:`None`)
        z_dim (int, optional): The dimension of node-level representations. 
            Required if :obj:`node_level` = :obj:`True`. (default: :obj:`None`)
        proj (string, or Module, optional): Projection head for graph-level representation. 
            If string, should be one of :obj:`"linear"` or :obj:`"MLP"`. Required if
            :obj:`graph_level` = :obj:`True`. (default: :obj:`None`)
        proj_n (string, or Module, optional): Projection head for node-level representations. 
            If string, should be one of :obj:`"linear"` or :obj:`"MLP"`. Required if
            :obj:`node_level` = :obj:`True`. (default: :obj:`None`)
        neg_by_crpt (bool, optional): The mode to obtain negative samples in JSE. If True, 
            obtain negative samples by performing corruption. Otherwise, consider pairs of
            different graph samples as negative pairs. Only used when 
            :obj:`objective` = :obj:`"JSE"`. (default: :obj:`False`)
        tau (int): The tempurature parameter in InfoNCE (NT-XENT) loss. Only used when 
            :obj:`objective` = :obj:`"NCE"`. (default: :obj:`0.5`)
        device (int, or `torch.device`, optional): The device to perform computation.
        choice_model (string, optional): Whether to yield model with :obj:`best` training loss or
            at the :obj:`last` epoch. (default: :obj:`last`)
        model_path (string, optinal): The directory to restore the saved model. 
            (default: :obj:`models`)
    """
    
    def __init__(self, objective, views_fn,
                 graph_level=True,
                 node_level=False,
                 z_dim=None,
                 z_n_dim=None,
                 proj=None,
                 proj_n=None,
                 neg_by_crpt=False,
                 tau=0.5,
                 device=None,
                 choice_model='last',
                 model_path='models'):

        assert node_level is not None or graph_level is not None
        assert not (objective=='NCE' and neg_by_crpt)

        super(Contrastive, self).__init__()
        self.loss_fn = self._get_loss(objective)
        self.views_fn = views_fn # fn: (batched) graph -> graph
        self.node_level = node_level
        self.graph_level = graph_level
        self.z_dim = z_dim
        self.z_n_dim = z_n_dim
        self.proj = proj
        self.proj_n = proj_n
        self.neg_by_crpt = neg_by_crpt
        self.tau = tau
        self.choice_model = choice_model
        self.model_path = model_path
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device
        
        
    def train(self, encoder, data_loader, optimizer, epochs, per_epoch_out=False):
        r"""Perform contrastive training and yield trained encoders per epoch or after
        the last epoch.
        
        Args:
            encoder (Module, or list of Module): A graph encoder shared by all views or a list 
                of graph encoders dedicated for each view. If :obj:`node_level` = :obj:`False`, 
                the encoder should return tensor of shape [:obj:`n_graphs`, :obj:`z_dim`].
                Otherwise, return tuple of shape ([:obj:`n_graphs`, :obj:`z_dim`], 
                [:obj:`n_nodes`, :obj:`z_n_dim`]) representing graph-level and node-level embeddings.
            dataloader (Dataloader): Dataloader for unsupervised learning or pretraining.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in encoder(s).
            epochs (int): Number of total training epochs.
            per_epoch_out (bool): If True, yield trained encoders per epoch. Otherwise, only yield
                the final encoder at the last epoch. (default: :obj:`False`)
                
        :rtype: :class:`generator`.
        """
        self.per_epoch_out = per_epoch_out
        
        if self.z_n_dim is None:
            self.proj_out_dim = self.z_dim
        else:
            self.proj_out_dim = self.z_n_dim
        
        if self.graph_level and self.proj is not None:
            self.proj_head_g = self._get_proj(self.proj, self.z_dim).to(self.device)
            optimizer.add_param_group({"params": self.proj_head_g.parameters()})
        elif self.graph_level:
            self.proj_head_g = lambda x: x
        else:
            self.proj_head_g = None
            
        if self.node_level and self.proj_n is not None:
            self.proj_head_n = self._get_proj(self.proj_n, self.z_n_dim).to(self.device)
            optimizer.add_param_group({"params": self.proj_head_n.parameters()})
        elif self.node_level:
            self.proj_head_n = lambda x: x
        else:
            self.proj_head_n = None
        
        if isinstance(encoder, list):
            encoder = [enc.to(self.device) for enc in encoder]
        else:
            encoder = encoder.to(self.device)
            
        if self.node_level and self.graph_level:
            train_fn = self.train_encoder_node_graph
        elif self.graph_level:
            train_fn = self.train_encoder_graph
        else:
            train_fn = self.train_encoder_node
            
        for enc in train_fn(encoder, data_loader, optimizer, epochs):
            yield enc

        
    def train_encoder_graph(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be Tensor for graph-level embedding
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
            [enc.train() for enc in encoders]
        else:
            encoder.train()
            encoders = [encoder]*len(self.views_fn)

        try:
            self.proj_head_g.train()
        except:
            pass
        
        min_loss = 1e9
        with trange(epochs) as t:
            for epoch in t:
                epoch_loss = 0.0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                for data in data_loader:
                    optimizer.zero_grad()
                    if None in self.views_fn: 
                        # For view fn that returns multiple views
                        views = []
                        for v_fn in self.views_fn:
                            if v_fn is not None:
                                views += [*v_fn(data)]
                    else:
                        views = [v_fn(data) for v_fn in self.views_fn]

                    zs = []
                    for view, enc in zip(views, encoders):
                        z = self._get_embed(enc, view.to(self.device))
                        zs.append(self.proj_head_g(z))

                    loss = self.loss_fn(zs, neg_by_crpt=self.neg_by_crpt, tau=self.tau)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    
                if self.per_epoch_out:
                    yield encoder, self.proj_head_g
                        
                t.set_postfix(loss='{:.6f}'.format(float(loss)))
                
                if self.choice_model == 'best' and epoch_loss < min_loss:
                    min_loss = epoch_loss

                    if not os.path.exists(self.model_path):
                        try:
                            os.mkdir(self.model_path)
                        except:
                            raise RuntimeError('cannot create model path')

                    if isinstance(encoder, list):
                        for i, enc in enumerate(encoder):
                            torch.save(enc.state_dict(), self.model_path+'/enc%d_best.pkl'%i)
                    else:
                        torch.save(encoder.state_dict(), self.model_path+'/enc_best.pkl')
            
            if self.choice_model == 'best':
                
                if not os.path.exists(self.model_path):
                    try:
                        os.mkdir(self.model_path)
                    except:
                        raise RuntimeError('cannot create model path')

                if isinstance(encoder, list):
                    for i, enc in enumerate(encoder):
                        enc.load_state_dict(torch.load(self.model_path+'/enc%d_best.pkl'%i))
                else:
                    encoder.load_state_dict(torch.load(self.model_path+'/enc_best.pkl'))

        if not self.per_epoch_out:
            yield encoder, self.proj_head_g

    
    def train_encoder_node(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be Tensor for node-level embedding
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
            [encoder.train() for encoder in encoders]
        else:
            encoder.train()
            encoders = [encoder]*len(self.views_fn)
        
        try:
            self.proj_head_n.train()
        except:
            pass
        
        min_loss = 1e9
        with trange(epochs) as t:
            for epoch in t:
                epoch_loss = 0.0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                for data in data_loader:
                    optimizer.zero_grad()
                    if None in self.views_fn:
                        # For view fn that returns multiple views
                        views = []
                        for v_fn in self.views_fn:
                            if v_fn is not None:
                                views += [*v_fn(data)]
                    else:
                        views = [v_fn(data) for v_fn in self.views_fn]

                    zs_n = []
                    for view, enc in zip(views, encoders):
                        z_n = self._get_embed(enc, view.to(self.device))
                        zs_n.append(self.proj_head_n(z_n))

                    loss = self.loss_fn(zs_g=None, zs_n=zs_n, batch=data.batch, 
                                        neg_by_crpt=self.neg_by_crpt, tau=self.tau)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    
                if self.per_epoch_out:
                    yield encoder, self.proj_head_n
                    
                t.set_postfix(loss='{:.6f}'.format(float(loss)))
                
                if self.choice_model == 'best' and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    if isinstance(encoder, list):
                        for i, enc in enumerate(encoder):
                            torch.save(enc.state_dict(), self.model_path+'/enc%d_best.pkl'%i)
                    else:
                        torch.save(encoder.state_dict(), self.model_path+'/enc_best.pkl')
            
            if self.choice_model == 'best':
                if isinstance(encoder, list):
                    for i, enc in enumerate(encoder):
                        enc.load_state_dict(torch.load(self.model_path+'/enc%d_best.pkl'%i))
                else:
                    encoder.load_state_dict(torch.load(self.model_path+'/enc_best.pkl'))

        if not self.per_epoch_out:
            yield encoder, self.proj_head_n
    
    
    def train_encoder_node_graph(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be tuple of (node_embed, graph_embed)
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
            [encoder.train() for encoder in encoders]
        else:
            encoder.train()
            encoders = [encoder]*len(self.views_fn)
        
        try:
            self.proj_head_n.train()
            self.proj_head_g.train()
        except:
            pass
        min_loss = 1e9
        with trange(epochs) as t:
            for epoch in t:
                epoch_loss = 0.0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                for data in data_loader:
                    optimizer.zero_grad()
                    if None in self.views_fn:
                        views = []
                        for v_fn in self.views_fn:
                            # For view fn that returns multiple views
                            if v_fn is not None:
                                views += [*v_fn(data)]
                        assert len(views)==len(encoders)
                    else:
                        views = [v_fn(data) for v_fn in self.views_fn]

                    zs_n, zs_g = [], []
                    for view, enc in zip(views, encoders):
                        z_g, z_n = self._get_embed(enc, view.to(self.device))
                        zs_n.append(self.proj_head_n(z_n))
                        zs_g.append(self.proj_head_g(z_g))

                    loss = self.loss_fn(zs_g, zs_n=zs_n, batch=data.batch, 
                                        neg_by_crpt=self.neg_by_crpt, tau=self.tau)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    
                if self.per_epoch_out:
                    yield encoder, (self.proj_head_g, self.proj_head_n)
                        

                t.set_postfix(loss='{:.6f}'.format(float(loss)))
                
                if self.choice_model == 'best' and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    if isinstance(encoder, list):
                        for i, enc in enumerate(encoder):
                            torch.save(enc.state_dict(), self.model_path+'/enc%d_best.pkl'%i)
                    else:
                        torch.save(encoder.state_dict(), self.model_path+'/enc_best.pkl')
            
            if self.choice_model == 'best' and not self.per_epoch_out:
                if isinstance(encoder, list):
                    for i, enc in enumerate(encoder):
                        enc.load_state_dict(torch.load(self.model_path+'/enc%d_best.pkl'%i))
                else:
                    encoder.load_state_dict(torch.load(self.model_path+'/enc_best.pkl'))

        if not self.per_epoch_out:
            yield encoder, (self.proj_head_g, self.proj_head_n)
    
    def _get_embed(self, enc, view):
        
        if self.neg_by_crpt:
            view_crpt = self._corrupt_graph(view)
            if self.node_level and self.graph_level:
                z_g, z_n = enc(view)
                z_g_crpt, z_n_crpt = enc(view_crpt)
                z = (torch.cat([z_g, z_g_crpt], 0),
                     torch.cat([z_n, z_n_crpt], 0))
            else:
                z = enc(view)
                z_crpt = enc(view_crpt)
                z = torch.cat([z, z_crpt], 0)
        else:
            z = enc(view)
        
        return z
                
    
    def _corrupt_graph(self, view):
        
        data_list = view.to_data_list()
        crpt_list = []
        for data in data_list:
            n_nodes = data.x.shape[0]
            perm = torch.randperm(n_nodes).long()
            crpt_x = data.x[perm]
            crpt_list.append(Data(x=crpt_x, edge_index=data.edge_index))
        view_crpt = Batch.from_data_list(crpt_list)

        return view_crpt
        
    
    def _get_proj(self, proj_head, in_dim):
        
        if callable(proj_head):
            return proj_head
        
        assert proj_head in ['linear', 'MLP']
        
        out_dim = self.proj_out_dim
        
        if proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
        elif proj_head == 'MLP':
            proj_nn = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(out_dim, out_dim))
            for m in proj_nn.modules():
                self._weights_init(m)
            
        return proj_nn
        
    def _weights_init(self, m):        
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def _get_loss(self, objective):
        
        if callable(objective):
            return objective
        
        assert objective in ['JSE', 'NCE']
        
        return {'JSE':JSE_loss, 'NCE':NCE_loss}[objective]
