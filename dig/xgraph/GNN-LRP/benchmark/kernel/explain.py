"""
FileName: explain.py
Description: 
Time: 2020/8/11 15:39
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
from benchmark.args import x_args
from benchmark.kernel.utils import Metric
from benchmark.kernel.evaluation import acc_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
from torch import Tensor
import copy
import time




class XCollector(object):

    def __init__(self, model, loader):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []
        self.hard_masks: Union[List, List[List[Tensor]]] = []

        self.__fidelity, self.__regular_fidelity, self.__contrastivity, self.__sparsity, self.__infidelity, self.__regular_infidelity = None, None, None, None, None, None
        self.__score = None
        self.model, self.loader = model, loader

    @property
    def maskout_preds(self) -> list:
        return self.__related_preds.get('maskout')

    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []
        self.hard_masks: Union[List, List[List[Tensor]]] = []

        self.__fidelity, self.__regular_fidelity, self.__contrastivity, self.__sparsity, self.__infidelity, self.__regular_infidelity = None, None, None, None, None, None
        self.__score = None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int) -> None:

        if self.__fidelity or self.__contrastivity or self.__sparsity or self.__infidelity or self.__regular_fidelity or self.__regular_infidelity:
            self.__fidelity, self.__regular_fidelity, self.__contrastivity, self.__sparsity, self.__infidelity, self.__regular_infidelity = None, None, None, None, None, None
            print(f'#W#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)

            self.__targets.append(label)
            self.masks.append(masks)

            # make the masks binary
            hard_masks = copy.deepcopy(masks)

            for i, hard_mask in enumerate(hard_masks):
                hard_mask[hard_mask >= 0] = 1
                hard_mask[hard_mask < 0] = 0

                hard_mask = hard_mask.to(torch.int64)

            self.hard_masks.append(hard_masks)

    @property
    def score(self):
        if self.__score:
            return self.__score
        else:

            self.__score = acc_score(self.model, self.loader)
            return self.__score

    @property
    def infidelity(self):
        if self.__infidelity:
            return self.__infidelity
        else:
            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            # Do Shapley Algorithm
            # mask_out_contribution = ((mask_out_preds - zero_mask_preds) + (one_mask_preds - masked_preds)) / 2
            contribution = one_mask_preds - masked_preds

            # for negative prediction's mask, the negative contribution is better.
            # So we multiply it with -1 to make it positive
            # target_ = torch.tensor(self.__targets)
            # target_[target_ == 0] = -1
            #
            # Discard above scripts:
            # because __related_pred only contains the prediction
            # probabilities of the correct labels. Thus higher is better.
            #
            # self.__Infidelity = (masked_contribution - mask_out_contribution).mean().item()
            self.__infidelity = contribution.mean().item()

            return self.__infidelity

    @property
    def regular_infidelity(self):
        if self.__regular_infidelity:
            return self.__regular_infidelity
        else:
            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            # Do Shapley Algorithm
            # mask_out_contribution = ((mask_out_preds - zero_mask_preds) + (one_mask_preds - masked_preds)) / 2
            contribution = (one_mask_preds - masked_preds) / \
                           torch.exp(torch.abs(one_mask_preds - zero_mask_preds))

            # for negative prediction's mask, the negative contribution is better.
            # So we multiply it with -1 to make it positive
            # target_ = torch.tensor(self.__targets)
            # target_[target_ == 0] = -1
            #
            # Discard above scripts:
            # because __related_pred only contains the prediction
            # probabilities of the correct labels. Thus higher is better.
            #
            # self.__Infidelity = (masked_contribution - mask_out_contribution).mean().item()
            self.__regular_infidelity = contribution.mean().item()

            return self.__regular_infidelity


    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        else:

            # score of output deriving from masked inputs
            # try:
            #     # self.__related_preds contains the preds that are corresponding to the y_true,
            #     # so closer to 1 is better
            #     # maskout_score = accuracy_score(torch.ones(len(self.__targets)), torch.tensor(self.__related_preds['maskout']).round())
            #
            # except ValueError as e:
            #     logger.warning(e)

            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            drop_probability = one_mask_preds - mask_out_preds


            # score of origin model output
            # origin_score = self.score

            # self.__fidelity = origin_score - maskout_score    # higher is better
            self.__fidelity = drop_probability.mean().item()
            return self.__fidelity

    @property
    def regular_fidelity(self):
        if self.__regular_fidelity:
            return self.__regular_fidelity
        else:

            # score of output deriving from masked inputs
            # try:
            #     # self.__related_preds contains the preds that are corresponding to the y_true,
            #     # so closer to 1 is better
            #     # maskout_score = accuracy_score(torch.ones(len(self.__targets)), torch.tensor(self.__related_preds['maskout']).round())
            #
            # except ValueError as e:
            #     logger.warning(e)

            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            drop_probability = (one_mask_preds - mask_out_preds) / \
                               torch.exp(torch.abs(one_mask_preds - zero_mask_preds))

            # score of origin model output
            # origin_score = self.score

            # self.__fidelity = origin_score - maskout_score    # higher is better
            self.__regular_fidelity = drop_probability.mean().item()
            return self.__regular_fidelity

    @property
    def contrastivity(self):
        if self.__contrastivity:
            return self.__contrastivity
        else:
            contrastivity = []
            for i in range(len(self.hard_masks)):
                for cur_label in range(len(self.hard_masks[0])):
                    if cur_label == self.__targets[i]:
                        continue

                    distance_hamington = \
                        (self.hard_masks[i][self.__targets[i]] != self.hard_masks[i][cur_label]).int().sum().item()

                    union = \
                        ((self.hard_masks[i][self.__targets[i]] + self.hard_masks[i][cur_label]) > 0).int().sum().item()

                    if union == 0:
                        continue

                    contrastivity.append(distance_hamington / union)

            self.__contrastivity = np.mean(contrastivity)

            return self.__contrastivity

    @property
    def sparsity(self):
        if self.__sparsity:
            return self.__sparsity
        else:
            # sparsity = []
            # for i in range(len(self.hard_masks)):
            #     for cur_label in range(len(self.hard_masks[0])):
            #         union = (self.hard_masks[i][cur_label] > 0).int().sum().item()
            #         V = self.hard_masks[i][0].shape[0]
            #         if V == 0:
            #             continue
            #         sparsity.append(1 - union / V)
            #
            # self.__sparsity = np.mean(sparsity)
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
        ax.set_title(f'{x_args.explainer}\nF: {x_collector.fidelity:.4f}  I: {x_collector.infidelity:.4f}  S: {x_collector.sparsity:.4f}')
        if x_args.save_fig:
            from definitions import ROOT_DIR
            import os
            print('save fig as:', os.path.join(ROOT_DIR, 'visual_results', f'{explainer.__class__.__name__}.png'))
            plt.savefig(os.path.join(ROOT_DIR, 'visual_results', f'{explainer.__class__.__name__}.png'), dpi=300)
        else:
            plt.show()


