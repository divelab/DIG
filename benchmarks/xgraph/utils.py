import json
import os
import torch
import random
import numpy as np
from abc import ABC
from torch_geometric.data import Data


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def fix_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def perturb_input(data, hard_edge_mask, subset):
    """add 2 additional empty node into the motif graph"""
    num_add_node = 2
    num_perturb_graph = 10
    subgraph_x = data.x[subset]
    subgraph_edge_index = data.edge_index[:, hard_edge_mask]
    row, col = data.edge_index

    mappings = row.new_full((data.num_nodes,), -1)
    mappings[subset] = torch.arange(subset.size(0), device=row.device)
    subgraph_edge_index = mappings[subgraph_edge_index]

    subgraph_y = data.y[subset]

    num_node_subgraph = subgraph_x.shape[0]

    # add two nodes to the subgraph, the node features are all 0.1
    subgraph_x = torch.cat([subgraph_x,
                            torch.ones(2, subgraph_x.shape[1]).to(subgraph_x.device)],
                           dim=0)
    subgraph_y = torch.cat([subgraph_y,
                            torch.zeros(num_add_node).type(torch.long).to(subgraph_y.device)], dim=0)

    perturb_input_list = []
    for _ in range(num_perturb_graph):
        to_node = torch.randint(0, num_node_subgraph, (num_add_node,))
        frm_node = torch.arange(num_node_subgraph, num_node_subgraph + num_add_node, 1)
        add_edges = torch.cat([torch.stack([to_node, frm_node], dim=0),
                               torch.stack([frm_node, to_node], dim=0),
                               torch.stack([frm_node, frm_node], dim=0)], dim=1)
        perturb_subgraph_edge_index = torch.cat([subgraph_edge_index,
                                                 add_edges.to(subgraph_edge_index.device)], dim=1)
        perturb_input_list.append(Data(x=subgraph_x, edge_index=perturb_subgraph_edge_index, y=subgraph_y))

    return perturb_input_list


class Recorder(ABC):
    def __init__(self, recorder_filename):
        # init the recorder
        self.recorder_filename = recorder_filename
        if os.path.isfile(recorder_filename):
            with open(recorder_filename, 'r') as f:
                self.recorder = json.load(f)
        else:
            self.recorder = {}
            check_dir(os.path.dirname(recorder_filename))

    @classmethod
    def load_and_change_dict(cls, ori_dict, experiment_settings, experiment_data):
            key = experiment_settings[0]
            if key not in ori_dict.keys():
                ori_dict[key] = {}
            if len(experiment_settings) == 1:
                ori_dict[key] = experiment_data
            else:
                ori_dict[key] = cls.load_and_change_dict(ori_dict[key],
                                                         experiment_settings[1:],
                                                         experiment_data)
            return ori_dict

    def append(self, experiment_settings, experiment_data):
        ex_dict = self.recorder

        self.recorder = self.load_and_change_dict(ori_dict=ex_dict,
                                                  experiment_settings=experiment_settings,
                                                  experiment_data=experiment_data)

    def save(self):
        with open(self.recorder_filename, 'w') as f:
            json.dump(self.recorder, f, indent=2)
