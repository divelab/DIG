import torch
import torch.nn as nn
import torch.nn.functional as F
from .spherenet import SphereNet
from .net_utils import *
from .geometric_computing import *
from .att import MH_ATT


class SphGen(nn.Module):
    def __init__(self, cutoff, num_node_types, num_layers, 
        hidden_channels, int_emb_size, basis_emb_size, out_emb_channels,
        num_spherical, num_radial, num_flow_layers, deq_coeff=0.9, use_gpu=True, n_att_heads=4):
        super(SphGen, self).__init__()
        self.use_gpu = use_gpu
        self.num_node_types = num_node_types        
        
        self.feat_net = SphereNet(cutoff, num_node_types, num_layers, hidden_channels, int_emb_size, basis_emb_size, out_emb_channels, num_spherical, num_radial)
        node_feat_dim, dist_feat_dim, angle_feat_dim, torsion_feat_dim = hidden_channels * 2, hidden_channels * 2, hidden_channels * 3, hidden_channels * 4

        self.node_flow_layers = nn.ModuleList([ST_Net_Exp(node_feat_dim, num_node_types, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.dist_flow_layers = nn.ModuleList([ST_Net_Exp(dist_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.angle_flow_layers = nn.ModuleList([ST_Net_Exp(angle_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.torsion_flow_layers = nn.ModuleList([ST_Net_Exp(torsion_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.focus_mlp = MLP(hidden_channels)
        self.deq_coeff = deq_coeff

        self.node_att = MH_ATT(n_att_heads, q_dim=hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        self.dist_att = MH_ATT(n_att_heads, q_dim=hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        self.angle_att = MH_ATT(n_att_heads, q_dim=2*hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        self.torsion_att = MH_ATT(n_att_heads, q_dim=3*hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        if use_gpu:
            self.node_att, self.dist_att, self.angle_att, self.torsion_att = self.node_att.to('cuda'), self.dist_att.to('cuda'), self.angle_att.to('cuda'), self.torsion_att.to('cuda')

        if use_gpu:
            self.feat_net = self.feat_net.to('cuda')
            self.node_flow_layers = self.node_flow_layers.to('cuda')
            self.dist_flow_layers = self.dist_flow_layers.to('cuda')
            self.angle_flow_layers = self.angle_flow_layers.to('cuda')
            self.torsion_flow_layers = self.torsion_flow_layers.to('cuda')
            self.focus_mlp = self.focus_mlp.to('cuda')


    def forward(self, data_batch):
        z, pos, batch = data_batch['atom_type'], data_batch['position'], data_batch['batch']
        node_feat = self.feat_net(z, pos, batch)
        focus_score = self.focus_mlp(node_feat)

        new_atom_type, focus = data_batch['new_atom_type'], data_batch['focus']
        x_z = F.one_hot(new_atom_type, num_classes=self.num_node_types).float()
        x_z += self.deq_coeff * torch.rand(x_z.size(), device=x_z.device)

        local_node_type_feat, query_batch = node_feat[focus[:,0]], batch[focus[:,0]]
        global_node_type_feat = self.node_att(local_node_type_feat, node_feat, node_feat, query_batch, batch)
        node_type_feat = torch.cat((local_node_type_feat, global_node_type_feat), dim=-1)
        node_latent, node_log_jacob = flow_forward(self.node_flow_layers, x_z, node_type_feat)
        
        node_type_emb_block = self.feat_net.init_e.emb
        node_type_emb = node_type_emb_block(new_atom_type)
        node_emb = node_feat * node_type_emb[batch]
        
        c1_focus, c2_c1_focus = data_batch['c1_focus'], data_batch['c2_c1_focus']
        dist, angle, torsion = data_batch['new_dist'], data_batch['new_angle'], data_batch['new_torsion']

        local_dist_feat, dist_query_batch = node_emb[focus[:,0]], batch[focus[:,0]]
        global_dist_feat = self.dist_att(local_dist_feat, node_emb, node_emb, dist_query_batch, batch)
        local_angle_feat, angle_query_batch = torch.cat((node_emb[c1_focus[:,1]], node_emb[c1_focus[:,0]]), dim=1), batch[c1_focus[:,0]]
        global_angle_feat = self.angle_att(local_angle_feat, node_emb, node_emb, angle_query_batch, batch)
        local_torsion_feat, torsion_query_batch = torch.cat((node_emb[c2_c1_focus[:,2]], node_emb[c2_c1_focus[:,1]], node_emb[c2_c1_focus[:,0]]), dim=1), batch[c2_c1_focus[:,0]]
        global_torsion_feat = self.torsion_att(local_torsion_feat, node_emb, node_emb, torsion_query_batch, batch)
        dist_feat = torch.cat((local_dist_feat, global_dist_feat), dim=-1)
        angle_feat = torch.cat((local_angle_feat, global_angle_feat), dim=-1)
        torsion_feat = torch.cat((local_torsion_feat, global_torsion_feat), dim=-1)
            
        dist_latent, dist_log_jacob = flow_forward(self.dist_flow_layers, dist, dist_feat)
        angle_latent, angle_log_jacob = flow_forward(self.angle_flow_layers, angle, angle_feat)
        torsion_latent, torsion_log_jacob = flow_forward(self.torsion_flow_layers, torsion, torsion_feat)

        return (node_latent, node_log_jacob), focus_score, (dist_latent, dist_log_jacob), (angle_latent, angle_log_jacob), (torsion_latent, torsion_log_jacob)


    def generate(self, type_to_atomic_number, num_gen=100, temperature=[1.0, 1.0, 1.0, 1.0], min_atoms=2, max_atoms=35, focus_th=0.5):
        with torch.no_grad():
            if self.use_gpu:
                prior_node = torch.distributions.normal.Normal(torch.zeros([self.num_node_types]).cuda(), temperature[0] * torch.ones([self.num_node_types]).cuda())
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[1] * torch.ones([1]).cuda())
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[2] * torch.ones([1]).cuda())
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[3] * torch.ones([1]).cuda())
            else:
                prior_node = torch.distributions.normal.Normal(torch.zeros([self.num_node_types]), temperature[0] * torch.ones([self.num_node_types]))
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]), temperature[1] * torch.ones([1]))
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]), temperature[2] * torch.ones([1]))
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]), temperature[3] * torch.ones([1]))
            
            node_type_emb_block = self.feat_net.init_e.emb
            z = torch.ones([num_gen, 1], dtype=int)
            pos = torch.zeros([num_gen, 1, 3], dtype=torch.float32)
            focuses = torch.zeros([num_gen, 0], dtype=int)
            if self.use_gpu:
                z, pos, focuses = z.cuda(), pos.cuda(), focuses.cuda()
            out_dict = {}

            mask_index = lambda mask, p: p[mask].view(num_gen, -1, 3)
            feat_index = lambda node_id, f: f[torch.arange(num_gen), node_id]
            pos_index = lambda node_id, p: p[torch.arange(num_gen), node_id].view(num_gen,1,3)

            for i in range(max_atoms):
                batch = torch.arange(num_gen, device=z.device).view(num_gen, 1).repeat(1, i+1)
                if i == 0:
                    node_feat = node_type_emb_block(z.view(-1))
                elif i == 1:
                    node_feat = self.feat_net.dist_only_forward(z.view(-1), pos.view(-1,3), batch.view(-1))
                else:
                    node_feat = self.feat_net(z.view(-1), pos.view(-1,3), batch.view(-1))
                
                focus_score = self.focus_mlp(node_feat).view(num_gen, i+1)
                can_focus = torch.logical_and(focus_score < focus_th, z > 0)
                complete_mask = (can_focus.sum(dim=-1) == 0)

                if i > max(0, min_atoms-2) and torch.sum(complete_mask) > 0:
                    out_dict[i+1] = {}
                    out_node_types = z[complete_mask].view(-1, i+1).cpu().numpy()
                    out_dict[i+1]['_atomic_numbers'] = type_to_atomic_number[out_node_types]
                    out_dict[i+1]['_positions'] = pos[complete_mask].view(-1, i+1, 3).cpu().numpy()
                    out_dict[i+1]['_focus'] = focuses[complete_mask].view(-1, i).cpu().numpy()
                
                continue_mask = torch.logical_not(complete_mask)
                dirty_mask = torch.nonzero(torch.isnan(focus_score).sum(dim=-1))[:,0]
                if len(dirty_mask) > 0:
                    continue_mask[dirty_mask] = False
                dirty_mask = torch.nonzero(torch.isinf(focus_score).sum(dim=-1))[:,0]
                if len(dirty_mask) > 0:
                    continue_mask[dirty_mask] = False

                if torch.sum(continue_mask) == 0:
                    break
                
                node_feat = node_feat.view(num_gen, i+1, -1)
                num_gen = torch.sum(continue_mask).cpu().item()
                z, pos, can_focus, focuses = z[continue_mask], pos[continue_mask], can_focus[continue_mask], focuses[continue_mask]
                focus_node_id = torch.multinomial(can_focus.float(), 1).view(num_gen)
                node_feat = node_feat[continue_mask]

                latent_node = prior_node.sample([num_gen])
                local_node_type_feat, query_batch = feat_index(focus_node_id, node_feat), torch.arange(num_gen, device=node_feat.device)
                key_value_batch = torch.arange(num_gen, device=node_feat.device).view(num_gen, 1).repeat(1, i+1).view(-1)
                global_node_type_feat = self.node_att(local_node_type_feat, node_feat.view(num_gen*(i+1),-1), node_feat.view(num_gen*(i+1),-1), query_batch, key_value_batch)
                node_type_feat = torch.cat((local_node_type_feat, global_node_type_feat), dim=-1)

                latent_node = flow_reverse(self.node_flow_layers, latent_node, node_type_feat)
                node_type_id = torch.argmax(latent_node, dim=1)
                node_type_emb = node_type_emb_block(node_type_id)
                node_emb = node_feat * node_type_emb.view(num_gen, 1, -1)

                latent_dist = prior_dist.sample([num_gen])
                local_dist_feat = feat_index(focus_node_id, node_emb)
                global_dist_feat = self.dist_att(local_dist_feat, node_emb.view(num_gen*(i+1),-1), node_emb.view(num_gen*(i+1),-1), query_batch, key_value_batch)
                dist_feat = torch.cat((local_dist_feat, global_dist_feat), dim=-1)

                dist = flow_reverse(self.dist_flow_layers, latent_dist, dist_feat)
                # dist = dist.abs()
                if i == 0:
                    new_pos = torch.cat((dist, torch.zeros_like(dist, device=dist.device), torch.zeros_like(dist, device=dist.device)), dim=-1)
                if i > 0:
                    mask = torch.ones([num_gen, i+1], dtype=torch.bool)
                    mask[torch.arange(num_gen), focus_node_id] = False
                    c1_dists = torch.sum(torch.square(mask_index(mask, pos) - pos_index(focus_node_id, pos)), dim=-1)
                    c1_node_id = torch.argmin(c1_dists, dim=-1)
                    c1_node_id[c1_node_id >= focus_node_id] += 1
                    
                    latent_angle = prior_angle.sample([num_gen])
                    local_angle_feat = torch.cat((feat_index(focus_node_id, node_emb), feat_index(c1_node_id, node_emb)), dim=1)
                    global_angle_feat = self.angle_att(local_angle_feat, node_emb.view(num_gen*(i+1),-1), node_emb.view(num_gen*(i+1),-1), query_batch, key_value_batch)
                    angle_feat = torch.cat((local_angle_feat, global_angle_feat), dim=-1)

                    angle = flow_reverse(self.angle_flow_layers, latent_angle, angle_feat)
                    # angle = angle.abs()
                    if i == 1:
                        fc1 = feat_index(c1_node_id, pos) - feat_index(focus_node_id, pos)
                        new_pos_x = torch.cos(angle) * torch.sign(fc1[:,0:1]) * dist
                        new_pos_y = torch.sin(angle) * torch.sign(fc1[:,0:1]) * dist
                        new_pos = torch.cat((new_pos_x, new_pos_y, torch.zeros_like(dist, device=dist.device)), dim=-1)
                        new_pos += feat_index(focus_node_id, pos)
                    else:
                        mask[torch.arange(num_gen), c1_node_id] = False
                        c2_dists = torch.sum(torch.square(mask_index(mask, pos) - pos_index(c1_node_id, pos)), dim=-1)
                        c2_node_id = torch.argmin(c2_dists, dim=-1)
                        c2_node_id[c2_node_id >= torch.minimum(focus_node_id, c1_node_id)] += 1
                        c2_node_id[c2_node_id >= torch.maximum(focus_node_id, c1_node_id)] += 1
                        
                        latent_torsion = prior_torsion.sample([num_gen])
                        local_torsion_feat = torch.cat((feat_index(focus_node_id, node_emb), feat_index(c1_node_id, node_emb), feat_index(c2_node_id, node_emb)), dim=1)
                        global_torsion_feat = self.torsion_att(local_torsion_feat, node_emb.view(num_gen*(i+1),-1), node_emb.view(num_gen*(i+1),-1), query_batch, key_value_batch)
                        torsion_feat = torch.cat((local_torsion_feat, global_torsion_feat), dim=-1)

                        torsion = flow_reverse(self.torsion_flow_layers, latent_torsion, torsion_feat)
                        new_pos = dattoxyz(pos_index(focus_node_id, pos), pos_index(c1_node_id, pos), pos_index(c2_node_id, pos), dist, angle, torsion)
                        

                z = torch.cat((z, node_type_id[:, None]), dim=1)
                pos = torch.cat((pos, new_pos.view(num_gen, 1, 3)), dim=1)
                focuses = torch.cat((focuses, focus_node_id[:,None]), dim=1)

            return out_dict