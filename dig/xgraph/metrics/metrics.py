"""
FileName: metrics.py
Description: 
Time: 2021/2/22 14:00
Project: DIG
Author: Shurui Gui
"""

import torch


def fidelity(ori_probs: torch.Tensor, unimportant_probs: torch.Tensor) -> float:

    drop_probability = ori_probs - unimportant_probs

    return drop_probability.mean().item()


def infidelity(ori_probs: torch.Tensor, important_probs: torch.Tensor) -> float:

    drop_probability = ori_probs - important_probs

    return drop_probability.mean().item()