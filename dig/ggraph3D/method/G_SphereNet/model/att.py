import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter



class MH_ATT(nn.Module):
    def __init__(self, n_att_heads=4, q_dim=128, k_dim=128, v_dim=128, out_dim=128):
        super(MH_ATT, self).__init__()
        self.n_att_heads = n_att_heads
        self.d_k = out_dim // n_att_heads
        self.q_proj = nn.Linear(q_dim, out_dim, bias=True)
        self.k_proj = nn.Linear(k_dim, out_dim, bias=True)
        self.v_proj = nn.Linear(v_dim, out_dim, bias=True)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=True)
    
    def forward(self, query, key, value, query_batch, key_value_batch):
        query_proj = self.q_proj(query).view(-1, self.n_att_heads, self.d_k)
        key_proj = self.k_proj(key).view(-1, self.n_att_heads, self.d_k)
        value_proj = self.v_proj(value).view(-1, self.n_att_heads, self.d_k)
        
        n_querys = query_proj.shape[0]
        key_value_mask = (key_value_batch[:,None] == query_batch[None,:]).sum(dim=-1) > 0
        key_proj, value_proj = key_proj[key_value_mask], value_proj[key_value_mask]
        query_num_nodes = (key_value_batch[:,None] == query_batch[None,:]).sum(dim=0)
        query_proj = torch.repeat_interleave(query_proj, query_num_nodes, dim=0)
        
        scaled_dots = torch.sum(query_proj * key_proj, dim=-1) / torch.sqrt(torch.tensor(self.d_k, dtype=float))
        new_query_batch = torch.repeat_interleave(torch.arange(n_querys, device=query_num_nodes.device), query_num_nodes, dim=0)
        att_scores = softmax(scaled_dots, index=new_query_batch, num_nodes=n_querys)
        att_outs = scatter(value_proj * att_scores[:,:,None], new_query_batch, dim=0, dim_size=n_querys)
        outs = self.out_proj(att_outs.view(n_querys, self.d_k*self.n_att_heads))

        return outs