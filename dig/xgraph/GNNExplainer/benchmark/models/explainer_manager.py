"""
FileName: explainer_manager.py
Description: The controller for all explainer models.
Time: 2020/8/4 8:56
Project: GNN_benchmark
Author: Shurui Gui
"""

import torch_geometric.nn as gnn
import torch.nn as nn
from inspect import isclass
import benchmark.models.explainers as explainers
from benchmark.args import XArgs
from benchmark.data import data_args


def load_explainer(explainer_name: str, model: nn.Module, args: XArgs) -> explainers.ExplainerBase:
    classes = [x for x in dir(explainers) if isclass(getattr(explainers, x))]

    try:
        assert explainer_name in classes
    except AssertionError:
        print(f'#E#Given explainer name {explainer_name} doesn\'t exist in module '
              f'benchmark.models.explainers.')
        exit(1)

    explainer = getattr(explainers, explainer_name)(model, epochs=args.epoch, lr=args.lr,
                                                    explain_graph=data_args.model_level == 'graph',
                                                    molecule=data_args.dataset_type == 'mol')

    return explainer
