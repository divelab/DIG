import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar

from gnn_explain import gnn_explain



explainer = gnn_explain(6, 30,  1, 50)  ####arguments: (max_node, max_step, target_class, max_iters)

explainer.train()





