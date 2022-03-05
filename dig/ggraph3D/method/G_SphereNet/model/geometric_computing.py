# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import sqrt, pi as PI
from torch_geometric.nn import knn_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_nearst_node(pos, batch):
    j, i = knn_graph(pos, 1, batch)
    adj_nearest = SparseTensor(row=i, col=j, value=torch.ones_like(j, device=j.device), sparse_sizes=(batch.size(0), batch.size(0)))
    j2, i2 = knn_graph(pos, 2, batch)
    adj_nearest2 = SparseTensor(row=i2, col=j2, value=torch.ones_like(j2, device=j2.device), sparse_sizes=(batch.size(0), batch.size(0)))

    return adj_nearest, SparseTensor.from_dense(adj_nearest2.to_dense() - adj_nearest.to_dense())


def xyztoda(pos, edge_index, num_nodes):
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    return dist, angle, i, j, idx_kj, idx_ji


def xyztodat(pos, edge_index, num_nodes, batch):
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # |sin_angle| * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    adj_nearest, adj_nearest2 = get_nearst_node(pos, batch)
    
    adj_nearest_row = adj_nearest[idx_j]
    adj_nearest2_row = adj_nearest2[idx_j]
    idx_k_n = adj_nearest_row.storage.col()
    idx_k_n2 = adj_nearest2_row.storage.col()
    mask = idx_k_n == idx_i
    idx_k_n[mask] = idx_k_n2[mask]

    # Calculate torsions.
    pos_j0 = pos[idx_k] - pos[idx_j]
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k_n] - pos[idx_j]
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(pos_ji, pos_j0)
    plane2 = torch.cross(pos_ji, pos_jk)
    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    torsion = torch.atan2(b, a) # -pi to pi
    torsion[torsion<=0]+=2*PI # 0 to 2pi

    return dist, angle, torsion, i, j, idx_kj, idx_ji


def dattoxyz(f, c1, c2, d, angle, torsion):
    c1c2 = c2 - c1
    c1f = f - c1
    c1c3 = c1f * torch.sum(c1c2 * c1f, dim=-1, keepdim=True) / torch.sum(c1f * c1f, dim=-1, keepdim=True)
    c3 = c1c3 + c1

    c3c2 = c2 - c3
    c3c4_1 = c3c2 * torch.cos(torsion[:, :, None])
    c3c4_2 = torch.cross(c3c2, c1f) / torch.norm(c1f, dim=-1, keepdim=True) * torch.sin(torsion[:, :, None])
    c3c4 = c3c4_1 + c3c4_2

    new_pos = -c1f / torch.norm(c1f, dim=-1, keepdim=True) * d[:, :, None] * torch.cos(angle[:, :, None])
    new_pos += c3c4 / torch.norm(c3c4, dim=-1, keepdim=True) * d[:, :, None] * torch.sin(angle[:, :, None])
    new_pos += f

    return new_pos
