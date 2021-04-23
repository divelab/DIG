"""
FileName: initial.py
Description: initialization
Time: 2020/7/30 11:48
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
import random
from benchmark import TrainArgs


def init(args: TrainArgs):

    # Fix Random seed
    torch.manual_seed(args.random_seed)

    # Default state is a training state
    torch.enable_grad()

