from .Graphair import graphair,aug_module,GCN,GCN_Body,Classifier
from dig.fairgraph.dataset.fairgraph_dataset import POKEC, NBA
import torch

import time
import os

class run():
    r"""
    The base script for running different Graphair methods.
    """

    def __init__(self):
        pass

    def run(self,device,dataset,model='Graphair',epochs=10_000,test_epochs=1_000,batch_size=1_000,
            lr=1e-4,weight_decay=1e-5):
        r""" The run script for training and validation

        Args:
            device (torch.device): Device for computation.
            model (str, optional): Should be one of the Graphair, FairGraph, FairAug, GCA, Grace. Defaults to Graphair.
            dataset (str, optional): The dataset to train on. Should be one of POKEC or NBA.
            epochs (int, optional): Number of epochs to train on. Defaults to 10_000.
            batch_size (int, optional): Number of samples in each minibatch in the training. Defaults to 1_000.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay factor for regularization term. Defaults to 1e-5.
            save_dir (str, optional): The path to save trained models. If set to :obj:`''`, will not save the model. Defaults to ''.
            log_dir (str, optional): The path to save log files. If set to :obj:`''`, will not save the log files. Defaults to ''.
        """

        # Train script

        dataset_name = dataset.name

        features = dataset.features
        if dataset_name=='POKEC':
            minibatch = dataset.minibatch
        sens = dataset.sens
        adj = dataset.adj
        idx_sens = dataset.idx_sens_train

        # generate model
        if model=='Graphair':
            aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
            f_encoder = GCN_Body(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, dropout = 0.1, nlayer = 2).to(device)
            sens_model = GCN(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, nclass = 1).to(device)
            classifier_model = Classifier(input_dim=64,hidden_dim=64)
            model = graphair(aug_model=aug_model,f_encoder=f_encoder,sens_model=sens_model,classifier_model=classifier_model, lr=lr,weight_decay=weight_decay,batch_size=batch_size,dataset=dataset_name).to(device)
        
        if dataset_name=='POKEC':
            # call fit_batch_GraphSAINT
            st_time = time.time()
            model.fit_batch_GraphSAINT(epochs=epochs,adj=adj, x=features,sens=sens,idx_sens = idx_sens,minibatch=minibatch, warmup=0, adv_epoches=1)
            print("Training time: ", time.time() - st_time)

        
        if dataset_name=='NBA':
            # call fit_whole
            st_time = time.time()
            model.fit_whole(epochs=epochs,adj=adj, x=features,sens=sens,idx_sens = idx_sens,warmup=0, adv_epoches=1)
            print("Training time: ", time.time() - st_time)


        # Test script
        if model=='Graphair':
            aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
            f_encoder = GCN_Body(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, dropout = 0.1, nlayer = 2).to(device)
            sens_model = GCN(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, nclass = 1).to(device)
            classifier_model = Classifier(input_dim=64,hidden_dim=64)
            model = graphair(aug_model=aug_model,f_encoder=f_encoder,sens_model=sens_model,classifier_model=classifier_model,lr=lr,weight_decay=weight_decay,batch_size=batch_size,dataset=dataset_name).to(device)
        
        # call test
        model.test(adj=adj,features=features,labels=dataset.labels,epochs=test_epochs,idx_train=dataset.idx_train,idx_val=dataset.idx_val,idx_test=dataset.idx_test,sens=sens)

