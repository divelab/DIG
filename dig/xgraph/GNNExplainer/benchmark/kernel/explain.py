"""
FileName: explain.py
Description: 
Time: 2020/8/11 15:39
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
from benchmark.args import x_args
from definitions import ROOT_DIR
import os
from benchmark.kernel.utils import Metric
from benchmark.kernel.evaluation import acc_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
from torch import Tensor
import copy
import time
import metrics



class XCollector(object):

    def __init__(self, model, loader):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__fidelity, self.__sparsity = None, None
        self.__score = None
        self.model, self.loader = model, loader


    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__fidelity, self.__sparsity = None, None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int) -> None:

        if self.__fidelity or self.__sparsity:
            self.__fidelity, self.__sparsity = None, None
            print(f'#W#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)

            self.__targets.append(label)
            self.masks.append(masks)


    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        else:

            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            return metrics.fidelity(one_mask_preds, mask_out_preds)

    @property
    def sparsity(self):
        if self.__sparsity:
            return self.__sparsity
        else:
            self.__sparsity = x_args.sparsity
            return self.__sparsity




def sample_explain(explainer, data, x_collector: XCollector, **kwargs):
    data.to(x_args.device)
    node_idx = kwargs.get('node_idx')
    y_idx = 0 if node_idx is None else node_idx  # if graph level: y_idx = 0 if node level: y_idx = node_idx

    if torch.isnan(data.y[y_idx].squeeze()):
        return

    explain_tik = time.time()
    walks, masks, related_preds = \
        explainer(data.x, data.edge_index, **kwargs)
    explain_tok = time.time()
    print(f"#D#Explainer prediction time: {explain_tok - explain_tik:.4f}")

    x_collector.collect_data(masks,
                             related_preds,
                             data.y[y_idx].squeeze().long().item())

    if x_args.vis:
        gt_label = data.y[y_idx].squeeze().long().item()
        if x_args.walk:
            labeled_walks = walks
            labeled_walks['score'] = labeled_walks['score'][:, gt_label]
            ax, G = explainer.visualize_walks(node_idx=0 if node_idx is None else node_idx, edge_index=data.edge_index,
                                              walks=labeled_walks, edge_mask=masks[gt_label],
                                              y=data.x[:, 0] if node_idx is None else data.y, num_nodes=data.x.shape[0])
        else:
            ax, G = explainer.visualize_graph(node_idx=0 if node_idx is None else node_idx, edge_index=data.edge_index,
                                             edge_mask=masks[gt_label],
                                             y=data.x[:, 0] if node_idx is None else data.y, num_nodes=data.x.shape[0])
        ax.set_title(f'{x_args.explainer}\nF: {x_collector.fidelity:.4f}  S: {x_collector.sparsity:.4f}')
        if x_args.save_fig:
            print('save fig as:', os.path.join(ROOT_DIR, 'visual_results', f'{explainer.__class__.__name__}.png'))
            plt.savefig(os.path.join(ROOT_DIR, 'visual_results', f'{explainer.__class__.__name__}.png'), dpi=300)
        else:
            plt.show()


