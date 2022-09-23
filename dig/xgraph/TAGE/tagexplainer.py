from typing import Optional
from tqdm import tqdm
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader as tDataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric import utils

from rdkit import Chem
import matplotlib.pyplot as plt
import networkx as nx
from textwrap import wrap

from dig.sslgraph.method.contrastive.objectives import JSE_loss, NCE_loss


class Explainer(nn.Module):
	''' The parametric explainer takes node embeddings and condition vector as inputs, 
	and predicts edge importance scores. Constructed as a 2-layer MLP.
	Args:
		embed_dim: Integer. Dimension of node embeddings.
		graph_level: Boolean. Whether to explain a graph-level prediction task or 
		node-level prediction task.
		hidden_dim: Integer. Hidden dimension of the MLP in the explainer.
	'''

	def __init__(self, embed_dim: int, graph_level: bool, hidden_dim: int = 600):

		super(Explainer, self).__init__()

		self.embed_dims = embed_dim * (2 if graph_level else 3)
		self.cond_dims = embed_dim

		self.emb_linear1 = nn.Sequential(nn.Linear(self.embed_dims, hidden_dim), nn.ReLU())
		self.emb_linear2 = nn.Linear(hidden_dim, 1)

		self.cond_proj = nn.Sequential(nn.Linear(self.cond_dims, self.embed_dims), nn.ReLU())

	def forward(self, embed, cond):
		'''
		Args:
			embeds: Tensor of shape [n_edges, 2*embed_dim] or [n_edges, 3*embed_dim*].
			cond: Tensor of shape [1, embed_dim]. Condition vector.
		'''
		cond = self.cond_proj(cond)
		out = embed * cond
		out = self.emb_linear1(out)
		out = self.emb_linear2(out)
		return out


class KHopSampler(MessagePassing):
	''' A real-time sampler that samples k-hop ego networks surrounding the given seed node. 
	Used in node-level explanations for efficient computing.
	Args:
		k: Integer. Number of hops to sample.
	'''
	def __init__(self, k):
		super(KHopSampler, self).__init__(aggr='max', flow='source_to_target', node_dim=0)
		self.k = k

	def forward(self, edge_index, num_nodes, node_idx=None):
		'''
		Args:
			edge_index: Tensor. Edge indices of the full graph.
			num_nodes: Integer. Total number of nodes in the full graph.
			node_idx: Integer. Index of the selected center node. If :obj:`None`, return
				ego networks of every node.
		Returns:
			Boolean tensor of shape [num_nodes]. Indicating whether each node is selected
			in the ego network.
		'''
		if node_idx is None:
			S = torch.eye(num_nodes).to(edge_index.device)
		else:
			S = torch.zeros(num_nodes).to(edge_index.device)
			S = S.scatter_(0, node_idx.to(edge_index.device), 1.0)

		edge_index = utils.to_undirected(edge_index, num_nodes=len(S))
		edge_index, _ = utils.add_self_loops(edge_index, num_nodes=len(S))
		for it in range(self.k):
			S = self.propagate(edge_index, x=S)
		return S.bool()


class MLPExplainer(torch.nn.Module):
	''' Downstream MLP explainer based on gradient of output w.r.t. input embedding.
	Args:
		mlp_model: :obj:`torch.nn.Module` The downstream model to be explained.
		device: Torch CUDA device.
	'''

	def __init__(self, mlp_model, device):
		super(MLPExplainer, self).__init__()
		self.model = mlp_model.to(device)
		self.device = device

	def forward(self, embeds, mode='explain'):
		'''Returns probability by forward propagation or gradients by backward propagation
		based on the mode specified.
		'''
		embeds = embeds.detach().to(self.device)
		self.model.eval()
		if mode == 'explain':
			return self.get_grads(embeds)
		elif mode == 'pred':
			return self.get_probs(embeds)
		else:
			raise NotImplementedError

	def get_probs(self, embeds):
		logits = self.model(embeds)
		if logits.shape[1] == 1:
			probs = torch.sigmoid(logits)
			probs = torch.cat([1-probs, probs], 1)
		else:
			probs = F.softmax(logits, dim=-1)
		return probs

	def get_grads(self, embeds):
		optimizer = torch.optim.SGD([embeds.requires_grad_()], lr=0.01)
		optimizer.zero_grad()
		logits = self.model(embeds)
		max_logits, _ = logits.max(dim=-1)
		max_logits.sum().backward()
		grads = embeds.grad
		grads = grads/torch.abs(grads).mean()
		return F.relu(grads)

class TAGExplainer(nn.Module):
	''' The TAGExplainer that performs 2-stage explanations. Includes training and inference.
	Args:
		model: :obj:`torch.nn.Module`. the GNN embedding model to be explained.
		embed_dim: Integer. Dimension of node embeddings.
		device: Torch CUDA device.
		explain_graph: Boolean. Whether to explain a graph-level prediction task or 
			node-level prediction task.
		coff_size, coff_ent: Hyper-parameters for mask regularizations.
		grad_scale: Float. The scale parameter for generating random condition vectors.
		loss_type: String from "NCE" or "JSE". Type of the contrastive loss.
	'''
	def __init__(self, model, embed_dim: int, device, explain_graph: bool = True, 
		coff_size: float = 0.01, coff_ent: float = 5e-4, grad_scale: float = 0.25,
		loss_type = 'NCE', t0: float = 5.0, t1: float = 1.0, num_hops: Optional[int] = None):

		super(TAGExplainer, self).__init__()
		self.device = device
		self.embed_dim = embed_dim
		self.explain_graph = explain_graph
		self.model = model.to(device)
		self.explainer = Explainer(embed_dim, explain_graph).to(device)

		# objective parameters for PGExplainer
		self.grad_scale = grad_scale
		self.coff_size = coff_size
		self.coff_ent = coff_ent
		self.t0 = t0
		self.t1 = t1
		self.loss_type = loss_type

		self._set_hops(num_hops)
		self.sampler = KHopSampler(self.num_hops)
		self.S = None


	def _set_hops(self, num_hops: int):
		if num_hops is None:
			self.num_hops = sum(
				[isinstance(m, MessagePassing) for m in self.model.modules()])
		else:
			self.num_hops = num_hops


	def __set_masks__(self, edge_mask: Tensor):
		""" Set the edge weights before message passing
		Args:
			edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
			  (default: :obj:`None`)
		"""
		edge_mask = edge_mask.to(self.device)
		for module in self.model.modules():
			if isinstance(module, MessagePassing):
				module.__explain__ = True
				module.__edge_mask__ = edge_mask


	def __clear_masks__(self):
		""" clear the edge weights to None, and set the explain flag to :obj:`False` """
		for module in self.model.modules():
			if isinstance(module, MessagePassing):
				module.__explain__ = False
				module.__edge_mask__ = None


	def __loss__(self, embed: Tensor, pruned_embed: Tensor, 
		condition: Tensor, edge_mask: Tensor, **kwargs):
		'''
		embed: Tensor of shape [n_sample, embed_dim]
		pruned_embed: Tensor of shape [n_sample, embed_dim]
		condition: Tensor of shape [1, embed_dim]
		'''
		max_items = kwargs.get('max_items')
		if self.loss_type=='NCE':
			contrast_loss = NCE_loss([condition*embed, condition*pruned_embed])
		elif max_items and len(embed) > max_items:
			contrast_loss = self.__batched_JSE__(condition*embed, condition*pruned_embed, max_items)
		else:
			contrast_loss = JSE_loss([condition*embed, condition*pruned_embed])

		size_loss = self.coff_size * torch.mean(edge_mask)
		edge_mask = edge_mask * 0.99 + 0.005
		mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
		mask_ent = self.coff_ent * torch.mean(mask_ent)

		loss = contrast_loss + size_loss + mask_ent
		return loss


	def __batched_JSE__(self, cond_embed, cond_pruned_embed, batch_size):
		loss = 0
		for i, (z1, z2) in enumerate(tDataLoader(
			TensorDataset(cond_embed, cond_pruned_embed), batch_size)):
			if len(z1)<=1:
				i -= 1
				break
			loss += JSE_loss([z1, z2])
		return loss/(i+1.0)

	def __rand_cond__(self, n_sample, max_val=None):
		lap = torch.distributions.laplace.Laplace(loc=0, scale=self.grad_scale)
		cond = F.relu(lap.sample([n_sample, self.embed_dim])).to(self.device)
		if max_val is not None:
			cond = torch.clip(cond, max=max_val)
		return cond

	def get_subgraph(self, node_idx: int, data: Data):

		x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
		num_nodes, num_edges = x.size(0), edge_index.size(1)
		col, row = edge_index
		
		node_mask = self.sampler(edge_index, num_nodes, node_idx)
		edge_mask = node_mask[row] & node_mask[col]
		subset = torch.nonzero(node_mask).view(-1)
		edge_index, edge_attr = utils.subgraph(node_mask, edge_index, edge_attr, 
			relabel_nodes=True, num_nodes=num_nodes)

		x = x[subset]
		y = y[subset] if y is not None else None
		batch = batch[subset] if batch is not None else None

		data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)
		return data, subset


	def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
		""" Sample from the instantiation of concrete distribution when training """
		if training:
			random_noise = torch.rand(log_alpha.shape)
			random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
			gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
			gate_inputs = gate_inputs.sigmoid()
		else:
			gate_inputs = log_alpha

		return gate_inputs


	def explain(self, data: Data, embed: Tensor, condition: Tensor,
				tmp: float = 1.0, training: bool = False, **kwargs):
		"""
		explain the GNN behavior for graph with explanation network
		Args:
			x (:obj:`torch.Tensor`): Node feature matrix with shape
			  :obj:`[num_nodes, dim_node_feature]`
			edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
			  with shape :obj:`[2, num_edges]`
			embed (:obj:`torch.Tensor`): Node embedding matrix with shape :obj:`[num_nodes, dim_embedding]`
			tmp (:obj`float`): The temperature parameter fed to the sample procedure
			training (:obj:`bool`): Whether in training procedure or not
		Returns:
			probs (:obj:`torch.Tensor`): The classification probability for graph with edge mask
			edge_mask (:obj:`torch.Tensor`): The probability mask for graph edges
		"""
		
		nodesize = embed.shape[0]
		feature_dim = embed.shape[1]
		col, row = data.edge_index
		f1 = embed[col]
		f2 = embed[row]
		if self.explain_graph:
			f12self = torch.cat([f1, f2], dim=-1)
		else:
			node_idx = kwargs.get('node_idx')
			self_embed = embed[node_idx].repeat(f1.shape[0], 1)
			f12self = torch.cat([f1, f2, self_embed], dim=-1)

		# using the node embedding to calculate the edge weight
		h = self.explainer(f12self.to(self.device), condition.to(self.device))

		mask_val = h.reshape(-1)
		values = self.concrete_sample(mask_val, beta=tmp, training=training)
		try:
			out_log = '%.4f, %.4f, %.4f, %.4f'%(
                h.max().item(), values.max().item(), h.min().item(), values.min().item())
		except:
			out_log = ''
		mask_sparse = torch.sparse_coo_tensor(
			data.edge_index, values, (nodesize, nodesize)
		)
		mask_sigmoid = mask_sparse.to_dense()

		# set the symmetric edge weights
		sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
		edge_mask = sym_mask[col, row]

		# inverse the weights before sigmoid in MessagePassing Module
		inv_sigmoid = lambda x: torch.log(x/(1-x))
		self.__clear_masks__()
		self.__set_masks__(inv_sigmoid(edge_mask))

		# the model prediction with edge mask
		embed = self.model(data)

		self.__clear_masks__()
		return embed, edge_mask, out_log


	def train_explainer_graph(self, loader, lr=0.001, epochs=10):
		""" training the explanation network by gradient descent(GD) using Adam optimizer """
		optimizer = Adam(self.explainer.parameters(), lr=lr)
		for epoch in range(epochs):
			tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))
			self.model.eval()
			self.explainer.train()
			pbar = tqdm(loader)
			for data in pbar:
				optimizer.zero_grad()
				data = data.to(self.device)
				embed, node_embed = self.model(data, emb=True)
				cond = self.__rand_cond__(1)
				pruned_embed, mask, log = self.explain(data, embed=node_embed, 
					condition=cond, tmp=tmp, training=True)
				loss = self.__loss__(embed, pruned_embed, cond, mask)
				pbar.set_postfix({'loss': loss.item(), 'log': log})
				loss.backward()
				optimizer.step()

                
	def train_large_explainer_node(self, loader, batch_size=2, lr=0.001, epochs=10, max_items=2000):
		""" training the explanation network by gradient descent(GD) using Adam optimizer """
		optimizer = Adam(self.explainer.parameters(), lr=lr, weight_decay=0.01)
			# train the mask generator
		for epoch in range(epochs):
			self.model.eval()
			self.explainer.train()
			for dt_idx, data in enumerate(loader):
				loss = 0.0
				optimizer.zero_grad()
				tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))
				data.to(self.device)
				
				with torch.no_grad():
					try:
						data = Batch.from_data_list([data])
					except:
						pass
					try:
						mask = data.train_mask
					except:
						mask = torch.ones_like(data.batch).bool()
				
				node_batches = torch.utils.data.DataLoader(torch.where(mask)[0].tolist(),
					batch_size=batch_size, shuffle=True)
				pbar = tqdm(node_batches)
				for node_batch in pbar:
					cond = self.__rand_cond__(1)
					pruned_embeds, embeds = [], []
					for node_idx in node_batch:
						subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data)
						if subgraph.edge_index.shape[0]>10000 or subgraph.x.shape[0]>3000 or subgraph.x.shape[0]<2:
							continue
						new_node_idx = int(torch.where(subset == node_idx)[0])
						with torch.no_grad():
							subg_embeds = self.model(subgraph)
						pruned_embed, mask, log = self.explain(subgraph, subg_embeds.to(self.device), 
							condition=cond, tmp=tmp, training=True, node_idx=new_node_idx)
						embeds.append(subg_embeds.cpu())#[new_node_idx:new_node_idx+1])
						pruned_embeds.append(pruned_embed.cpu())#[new_node_idx:new_node_idx+1])
					embeds = torch.cat(embeds, 0).to(self.device)
					if len(embeds) <= 1:
						continue
					pruned_embeds = torch.cat(pruned_embeds, 0).to(self.device)
					loss = self.__loss__(embeds, pruned_embeds, cond, mask)#, max_items=2000)
					if torch.isnan(loss):
						continue
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.explainer.parameters(), 2.0)
					optimizer.step()
					pbar.set_postfix({'loss': loss.item(), 'log': log})
                

	def train_explainer_node(self, loader, batch_size=128, lr=0.001, epochs=10):
		""" training the explanation network by gradient descent(GD) using Adam optimizer """
		optimizer = Adam(self.explainer.parameters(), lr=lr)
			# train the mask generator
		for epoch in range(epochs):
			self.model.eval()
			self.explainer.train()
			for dt_idx, data in enumerate(loader):
				loss = 0.0
				optimizer.zero_grad()
				tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))
				with torch.no_grad():
					self.model.cpu()
					all_embeds = self.model(data)
				self.model.to(self.device)
				data.to(self.device)
				try:
					mask = data.train_mask
				except:
					mask = torch.ones_like(data.batch).bool()

				node_batches = torch.utils.data.DataLoader(torch.where(mask)[0].tolist(),
					batch_size=batch_size, shuffle=True)
				pbar = tqdm(node_batches)
				for node_batch in pbar:
					cond = self.__rand_cond__(1)
					embeds = all_embeds[node_batch].to(self.device)
					pruned_embeds = []
					masks = []
					for node_idx in node_batch:
						subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data)
						new_node_idx = int(torch.where(subset == node_idx)[0])
						pruned_embed, mask, log = self.explain(subgraph, all_embeds[subset].to(self.device), 
							condition=cond, tmp=tmp, training=True, node_idx=new_node_idx)
						pruned_embeds.append(pruned_embed.cpu()[new_node_idx:new_node_idx+1])
						masks.append(mask)
					pruned_embeds = torch.cat(pruned_embeds, 0).to(self.device)
					masks = torch.cat(masks, 0)
					if len(pruned_embeds)<=1:
						continue
					loss = self.__loss__(embeds, pruned_embeds, cond, masks)
					loss.backward()
					optimizer.step()
					pbar.set_postfix({'loss': loss.item(), 'log': log})


	def __edge_mask_to_node__(self, data, edge_mask, top_k):
		threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
		hard_mask = (edge_mask > threshold).cpu()
		edge_idx_list = torch.where(hard_mask == 1)[0]

		selected_nodes = []
		edge_index = data.edge_index.cpu().numpy()
		for edge_idx in edge_idx_list:
			selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
		selected_nodes = list(set(selected_nodes))
		maskout_nodes = [node for node in range(data.x.shape[0]) if node not in selected_nodes]

		node_mask = torch.zeros(data.num_nodes).type(torch.float32).to(self.device)
		node_mask[maskout_nodes] = 1.0
		return node_mask


	def forward(self, data: Data, mlp_explainer: nn.Module, **kwargs):
		""" explain the GNN behavior for graph and calculate the metric values.
		The interface for the :class:`dig.evaluation.XCollector`.

		Args:
			x (:obj:`torch.Tensor`): Node feature matrix with shape
			  :obj:`[num_nodes, dim_node_feature]`
			edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
			  with shape :obj:`[2, num_edges]`
			kwargs(:obj:`Dict`):
			  The additional parameters
				- top_k (:obj:`int`): The number of edges in the final explanation results
				- y (:obj:`torch.Tensor`): The ground-truth labels

		:rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
		"""
		top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10
		node_idx = kwargs.get('node_idx')
		cond_vec = kwargs.get('cond_vec')
		self.model.eval()
		mlp_explainer = mlp_explainer.to(self.device).eval()
		data = data.to(self.device)

		self.__clear_masks__()
		if node_idx is not None:
			node_embed = self.model(data)
			embed = node_embed[node_idx:node_idx+1]
		elif self.explain_graph:
			embed, node_embed = self.model(data, emb=True)
		else:
			assert node_idx is not None, "please input the node_idx"
		probs = mlp_explainer(embed, mode='pred')        
		grads = mlp_explainer(embed, mode='explain') if cond_vec is None else cond_vec
		probs = probs.squeeze()

		if self.explain_graph:
			subgraph = None
			target_class = torch.argmax(probs) if data.y is None else max(data.y.long(), 0) # sometimes labels are +1/-1
			_, edge_mask, log = self.explain(data, embed=node_embed, condition=grads, tmp=1.0, training=False)
			node_mask = self.__edge_mask_to_node__(data, edge_mask, top_k)
			masked_data = mask_fn(data, node_mask)
			masked_embed = self.model(masked_data)
			masked_prob = mlp_explainer(masked_embed, mode='pred')
			masked_prob = masked_prob[:, target_class]
			sparsity_score = sum(node_mask) / data.num_nodes
		else:
			target_class = torch.argmax(probs) if data.y is None else max(data.y[node_idx].long(), 0) # sometimes labels are +1/-1
			subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data)
			new_node_idx = torch.where(subset == node_idx)[0]
			_, edge_mask, log = self.explain(subgraph, node_embed[subset], condition=grads, 
				tmp=1.0, training=False, node_idx=new_node_idx)
			node_mask = self.__edge_mask_to_node__(subgraph, edge_mask, top_k)
			masked_embed = self.model(mask_fn(subgraph, node_mask))
			masked_prob = mlp_explainer(masked_embed, mode='pred')[new_node_idx, target_class.long()]
			sparsity_score = sum(node_mask) / subgraph.num_nodes

		# return variables
		pred_mask = edge_mask.detach().cpu()
		related_preds = [{
			'maskout': masked_prob.item(),
			'origin': probs[target_class].item(),
			'sparsity': sparsity_score}]
		return subgraph, pred_mask, related_preds


def mask_fn(data: Data, node_mask: np.array):
	""" subgraph building through spliting the selected nodes from the original graph """
	row, col = data.edge_index
	edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
	ret_edge_index = data.edge_index[:, edge_mask]
	ret_edge_attr = None if data.edge_attr is None else data.edge_attr[edge_mask] 
	data = Data(x=data.x, edge_index=ret_edge_index, 
		edge_attr=ret_edge_attr, batch=data.batch)
	return data
