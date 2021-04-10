import torch
import numpy as np
from model import GraphFlowModel, GraphFlowModel_rl
from config.prop_optim_config import conf
from prop_optim import PropOptim

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os


out_path = 'optim/'
if not os.path.isdir(out_path):
    os.mkdir(out_path)
runner = PropOptim(conf, out_path)
runner.reinforce()
