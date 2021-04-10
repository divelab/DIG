import torch
import numpy as np
from model import GraphFlowModel, GraphFlowModel_rl
from config.dense_gen_config import conf
from dense_gen import DensityGen

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os


out_path = 'density/'
if not os.path.isdir(out_path):
    os.mkdir(out_path)
runner = DensityGen(conf,'../datasets/zinc250k.csv', out_path)
runner.train()
