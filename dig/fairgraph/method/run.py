from Graphair import aug_module
from Graphair.graphair import Graphair
from Graphair.aug_module import aug_module
from Graphair.GCN import GCN_Body,GCN
from dig.fairgraph.dataset.fairgraph_dataset import POKEC
from torch_geometric.utils import to_scipy_sparse_matrix
from Graphair.graphsaint.minibatch import Minibatch
import numpy as np
import time
import os

class run():
    r"""
    The base script for running different Graphair methods.
    """

    def __init__(self):
        pass

    def train(self):
        r"""
        The script for training.
        
        Args:

        Returns:
        """
        pass

    def validation(self):
        r"""
        The Script for validation.
        
        Args:

        Returns:
        """
        pass

    def run(self,device,model='Graphair',dataset='POKEC',epochs=10_000,batch_size=1_000,
            lr=1e-4,weight_decay=1e-5,save_dir='',log_dir=''):
        r""" The run script for training and validation

        Args:
            device (torch.device): Device for computation.
            model (str, optional): Should be one of the Graphair, FairGraph, FairAug, GCA, Grace. Defaults to Graphair.
            dataset (str, optional): The dataset to train on. Should be one of POKEC or NBA. Defaults to POKEC.
            epochs (int, optional): Number of epochs to train on. Defaults to 10_000.
            batch_size (int, optional): Number of samples in each minibatch in the training. Defaults to 1_000.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay factor for regularization term. Defaults to 1e-5.
            save_dir (str, optional): The path to save trained models. If set to :obj:`''`, will not save the model. Defaults to ''.
            log_dir (str, optional): The path to save log files. If set to :obj:`''`, will not save the log files. Defaults to ''.
        """
        if save_dir !='':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        # make train data first
        if dataset=='POKEC':
            data = POKEC()
            data.describe() # Print a description of the data that we are going to use
            data = data[0] # since only one graph
            adj = to_scipy_sparse_matrix(data.edge_index)
            features = data.x
            labels = data.y
            sens = labels['I_am_working_in_field']
            idx_sens = None
            # Use minibatch from GraaphSAINT
            ids = np.arange(features.shape[0])
            role = {'tr':ids.copy(), 'va': ids.copy(), 'te':ids.copy()}
            train_params = {'sample_coverage': 500}
            train_phase = {'sampler': 'rw', 'num_root': batch_size, 'depth': 3, 'end':30}
            minibatch = Minibatch(adj, adj,role, train_params)
            minibatch.set_sampler(train_phase)
            pass

        # generate model
        if model=='Graphair':
            aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
            f_encoder = GCN_Body(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, dropout = 0.1, nlayer = 2).to(device)
            sens_model = GCN(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, nclass = 1).to(device)
            model = Graphair(aug_model=aug_model,f_encoder=f_encoder,sens_model=sens_model,lr=lr,weight_decay=weight_decay,batch_size=batch_size).to(device)
        
        # call fit_batch_GraphSAINT
        st_time = time.time()
        model.fit_batch_GraphSAINT(epochs=epochs,adj=adj, x=features,sens=sens,idx_sens = idx_sens,minibatch=minibatch, warmup=0, adv_epoches=1)
        print("Training time: ", time.time() - st_time)

        
        pass