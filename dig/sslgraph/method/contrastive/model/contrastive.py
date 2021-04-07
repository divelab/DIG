import sys, torch
from tqdm import trange
import torch.nn as nn
from torch_geometric.data import Batch, Data
from sslgraph.contrastive.objectives import NCE_loss, JSE_loss

class Contrastive(nn.Module):
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
        """
        Args:
            objective: String or function. If string, should be one of 'NCE' and 'JSE'.
            views_fn: List of functions. Functions to perform view transformation.
            graph_level: Boolean. Whether to include graph-level embedding for contrast.
            node_level: Boolean. Whether to include node-level embedding for contrast.
            proj: String, function or None. Projection head for graph-level representation. 
                If string, should be one of 'linear' and 'MLP'.
            proj_n: String, function or None. Projection head for node-level representations. 
                If string, should be one of 'linear' and 'MLP'. Required when node_level
                is True.
            neg_by_crpt: Boolean. If True, obtain negative samples by performing corruption.
                Otherwise, consider pairs of different graph samples as negative pairs.
                Should always be False when using InfoNCE objective.
        """
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
        """
        Args:
            encoder: Trainable pytorch model or list of models. Callable with inputs of Data 
                objects. If node_level is False, return tensor of shape [n_graphs, z_dim]. 
                Else, return tuple of shape ([n_graphs, z_dim], [n_nodes, z'_dim]) representing 
                graph-level and node-level embeddings.
            dataloader: Dataloader for contrastive training.
            optimizer: Torch optimizer.
            epochs: Integer.
            per_epoch_out: Boolean. If True, yield encoder per k epochs. Otherwise, only yield
                the final encoder at the last epoch.
        Returns:
            Generator that yield encoder per k epochs or the final encoder at the last epoch.
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

        self.proj_head_g.train()
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
        
        self.proj_head_n.train()
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
        
        self.proj_head_n.train()
        self.proj_head_g.train()
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
            return nn.Linear(in_dim, out_dim)
        elif proj_head == 'MLP':
            return nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(out_dim, out_dim))
        
        
    def _get_loss(self, objective):
        
        if callable(objective):
            return objective
        
        assert objective in ['JSE', 'NCE']
        
        return {'JSE':JSE_loss, 'NCE':NCE_loss}[objective]
