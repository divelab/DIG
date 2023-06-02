# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import torch
from torch_geometric.data import Batch, Data
from torch.nn import Linear, Sequential, ReLU, Softmax, BatchNorm1d as BN
from torch.distributions import Categorical
from .aug_encoder import AugEncoder
from .node_fm import NodeFM
from .node_drop import NodeDrop
from .edge_per import EdgePer
from ..constants import *


class Augmenter(torch.nn.Module):
    AUG_STR_TO_CLASS = {
        'node_fm': NodeFM,
        'node_drop': NodeDrop,
        'edge_per': EdgePer
    }

    def __init__(self, in_dim=64, num_layers=3, hid_dim=64, conv_type=ConvType.GIN, max_num_aug=8, use_stop_aug=False, aug_type_param_dict={}, uniform=False, rnn_input=RnnInputType.VIRTUAL, **kwargs):
        super(Augmenter, self).__init__()
        aug_type_list = aug_type_param_dict.keys()
        self.num_aug_types = len(aug_type_list)
        self.max_num_aug = max_num_aug

        self.aug_enc = AugEncoder(in_dim, num_layers, hid_dim, use_virtual_node=True, conv_type=conv_type, **kwargs)
        
        assert rnn_input in [RnnInputType.VIRTUAL, RnnInputType.ONE_HOT]
        self.rnn_hid_dim = hid_dim
        self.rnn_input = rnn_input
        if rnn_input == RnnInputType.VIRTUAL:
            self.aug_type_rnn = torch.nn.GRU(hid_dim, hid_dim)
        elif rnn_input == RnnInputType.ONE_HOT:
            self.aug_type_rnn = torch.nn.GRU(len(aug_type_list), hid_dim)

        out_dim = len(aug_type_list) + 1 if use_stop_aug else len(aug_type_list)
        self.aug_type_mlp = Sequential(
            Linear(hid_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, out_dim),
            Softmax(dim=-1)
        )

        self.aug_trans_heads = torch.nn.ModuleList()
        for aug_type in aug_type_list:
            self.aug_trans_heads.append(self.AUG_STR_TO_CLASS[aug_type](uniform=uniform, **aug_type_param_dict[aug_type]))


    def augment(self, data):
        cur_data, log_likelihood = data, torch.tensor(0.0, device=data.x.device)
        rnn_hidden = torch.zeros([1, 1, self.rnn_hid_dim], device=data.x.device)
        aug_type = None
        for _ in range(self.max_num_aug):
            node_emb, virtual_emb = self.aug_enc(cur_data)
            if self.rnn_input == RnnInputType.VIRTUAL:
                _, rnn_hidden = self.aug_type_rnn(virtual_emb.unsqueeze(0), rnn_hidden)
            elif self.rnn_input == RnnInputType.ONE_HOT:
                last_type_one_hot = torch.zeros([1, self.num_aug_types]) if aug_type is None\
                    else torch.nn.functional.one_hot(torch.tensor([aug_type]), self.num_aug_types)
                _, rnn_hidden = self.aug_type_rnn(last_type_one_hot.unsqueeze(0), rnn_hidden)
            aug_type_probs = self.aug_type_mlp(rnn_hidden).squeeze()
            aug_type = Categorical(aug_type_probs).sample().item()

            if aug_type == self.num_aug_types:
                category_likelihood = torch.log(aug_type_probs[aug_type])
                log_likelihood += category_likelihood
                break
            
            aug_trans_net = self.aug_trans_heads[aug_type]
            cur_data, aug_likelihood = aug_trans_net(cur_data, node_emb)
            
            category_likelihood = torch.log(aug_type_probs[aug_type])
            log_likelihood += category_likelihood
            if aug_likelihood:
                log_likelihood += aug_likelihood
            
        return cur_data, log_likelihood


    def forward(self, data):
        if isinstance(data, Batch) and self.aug_trans_heads[0].training:
            data_list = [self.augment(d) for d in data.to_data_list()]
            new_data, log_likelihoods = zip(*data_list)
            try:
                new_data = Batch.from_data_list(new_data)
            except:
                print(new_data)
            return new_data, torch.stack(log_likelihoods, 0)
        
        if isinstance(data, Batch):
            data_list = [self.augment(d)[0] for d in data.to_data_list()]
            new_data = Batch.from_data_list(data_list)
            return new_data
            
        if isinstance(data, Data):
            return self.augment(data)