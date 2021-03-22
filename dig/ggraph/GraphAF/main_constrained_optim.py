import torch
import numpy as np
from model import GraphFlowModel, GraphFlowModel_con_rl
from config.con_optim_config import conf
from con_optim import ConOptim

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os


out_path = 'cons_optim/'
if not os.path.isdir(out_path):
    os.mkdir(out_path)
runner = ConOptim(conf, data_file='../datasets/zinc_800_graphaf.csv', out_path=out_path)
runner.reinforce()
