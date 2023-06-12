import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from dig.auggraph.method.SMixup.model.GCN import GCN
from dig.auggraph.method.SMixup.model.GIN import GIN
from dig.auggraph.method.SMixup.model.GraphMatching import GraphMatching
from dig.auggraph.method.SMixup.utils.sinkhorn import Sinkhorn
from dig.auggraph.method.SMixup.utils.utils import NormalizedDegree, triplet_loss
from dig.auggraph.dataset.aug_dataset import TripleSet

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch.nn.functional import softmax

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score

class smixup():
    r"""
    The S-Mixup from the `"Graph Mixup with Soft Alignments" <https://icml.cc/virtual/2023/poster/24930>`_ paper.
    
    Args:
        data_root_path (string): Directory where datasets are saved. 
        dataset (string): Dataset Name.
        conf (dict): Hyperparameters of the graph matching network which is used to compute the soft alignments.
    """
    
    def __init__(self, data_root_path, dataset, GMNET_conf):
        self._get_dataset(data_root_path, dataset)
        self.GMNET_conf = GMNET_conf
        self._get_GMNET(self.GMNET_conf['nlayers'], self.GMNET_conf['nhidden'], self.GMNET_conf['bs'], self.GMNET_conf['lr'], self.GMNET_conf['epochs'])
      
    def _get_GMNET(self, GMNET_nlayers, GMNET_hidden, GMNET_bs, GMNET_lr, GMNet_epochs):  
        conf = {}
        conf_dis_param = {}
        conf_dis_param['num_layers'] = GMNET_nlayers
        conf_dis_param['hidden'] = GMNET_hidden
        conf_dis_param['model_type'] = 'gmnet'
        conf_dis_param['pool_type'] = 'sum'
        conf_dis_param['fuse_type'] = 'abs_diff'
        conf['dis_param'] = conf_dis_param

        conf['batch_size'] = GMNET_bs
        conf['start_lr'] = GMNET_lr
        conf['factor'] = 0.5
        conf['patience'] = 5
        conf['min_lr'] = 0.0000001
        conf['pre_train_path'] = None
        conf['max_num_epochs'] = GMNet_epochs

        conf['dis_param']['in_dim'] = self.dataset[0].x.shape[1]
        
        self.GMNET = GraphMatching(**conf['dis_param'])
        
        
    def _get_dataset(self, data_root_path, dataset):
        if (dataset == 'IMDBB'):
            dataset = TUDataset(data_root_path, name='IMDB-BINARY', use_node_attr=True)
            num_cls = 2
        elif dataset == 'MUTAG':
            dataset = TUDataset(data_root_path, name="MUTAG", use_node_attr=True)
            num_cls = 2
        elif dataset == 'PROTEINS':
            dataset = TUDataset(data_root_path, name=dataset, use_node_attr=True)
            num_cls = 2
        elif dataset == 'REDDITB':
            dataset = TUDataset(data_root_path, name="REDDIT-BINARY", use_node_attr=True)
            num_cls = 2
        elif dataset == 'IMDBM':
            dataset = TUDataset(data_root_path, name='IMDB-MULTI', use_node_attr=True)
            num_cls = 3
        elif dataset == 'REDDITM5':
            dataset = TUDataset(data_root_path, name='REDDIT-MULTI-5K', use_node_attr=True)
            num_cls = 5
        elif dataset == 'REDDITM12':
            dataset = TUDataset(data_root_path, name='REDDIT-MULTI-12K', use_node_attr=True)
            num_cls = 11
        elif dataset == 'NCI1':
            dataset = TUDataset(data_root_path, name='NCI1', use_node_attr=True)
            num_cls = 2
            
        if dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
                
        self.dataset = dataset
        self.num_cls = num_cls
        
    
    def train_test(self, batch_size, cls_model, cls_nlayers, cls_hidden, cls_dropout, cls_lr, cls_epochs, alpha, ckpt_path, sim_method = 'cos'):
        r"""
        This method first train a GMNET and then use the GMNET to perform S-Mixup. 
        
        Args:
            batch_size (int): Batch size of training the classifier.
            cls_model (string): Use GCN or GIN as the backbone of the classifier. 
            cls_nlayers (int): Number of GNN layers of the classifier.
            cls_hidden (int): Number of hidden units of the classifier.
            cls_dropout (float): Dropout ratio of the classifier.
            cls_lr (float): Initial learning rate of training the classifier. 
            cls_epochs (int): Training epochs of the classifier.
            alpha (float): Mixup ratio.
            ckpt_path (string): Location for saving checkpoints. 
            sim_method (string): Similarity function used to compute the assignment matrix. (default: :obj:`cos`)
        """
        criterion = nn.CrossEntropyLoss()
        test_accs = []
        kf = KFold(n_splits=10, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(kf.split(list(range(len(self.dataset))))):
            train_idx, val_idx = train_test_split(train_idx, test_size=0.1)
            train_set, val_set, test_set = self.dataset[train_idx.tolist()], self.dataset[val_idx.tolist()], self.dataset[test_idx.tolist()]
            
            self.train_GMNET(train_set)
            
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 8)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers = 8)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 8)

            if (cls_model == 'GCN'):
                model = GCN(self.dataset[0].x.shape[1], self.num_cls, cls_nlayers, cls_hidden, cls_dropout)
            elif (cls_model == 'GIN'):
                model = GIN(self.dataset[0].x.shape[1], self.num_cls, cls_nlayers, cls_hidden, cls_dropout)
            model.cuda()
            
            optimizer = optim.Adam(model.parameters(),
                                lr=cls_lr, weight_decay=1e-5)
            
            best_acc = 0.0         
            for epoch in range(1, cls_epochs + 1):
                loss_accum = 0.0
                for step, batch in enumerate(train_loader):
                    batch = self.Mixup(batch, alpha = alpha, sim_method = sim_method)

                    model.train()
                    optimizer.zero_grad()

                    _, output = model(batch)
                    
                    loss = batch.lam * criterion(output, batch.y1.long()) + (1-batch.lam) * criterion(output, batch.y2.long())
                    loss.backward()
                    optimizer.step()

                    loss_accum += loss.item()

                train_loss = loss_accum / (step + 1)
                print("Epoch [{}] Train_loss {}".format(epoch, train_loss))

                y_label = []
                y_pred = []
                
                for step, batch in enumerate(val_loader):
                    batch = batch.cuda()
                    model.eval()
                    _, output = model(batch)                   
                    pred = torch.argmax(output, dim = 1).long()
                    y_pred = y_pred + pred.cpu().detach().numpy().flatten().tolist()
                    y = batch.y.long()
                    y_label = y_label + y.cpu().detach().numpy().flatten().tolist()

                acc_val = accuracy_score(y_pred, y_label)
                print("Epoch [{}] Test results:".format(epoch),
                        "acc_val: {:.4f}".format(acc_val),)
                
                if acc_val >= best_acc:
                    best_acc = acc_val
                    torch.save(model.state_dict(), ckpt_path + "/best_val.pth")
            
            model.load_state_dict(torch.load(ckpt_path + "/best_val.pth"))
            
            y_label = []
            y_pred = []
            for step, batch in enumerate(test_loader):
                batch = batch.cuda()
                model.eval()
                _, output = model(batch)
                pred = torch.argmax(output, dim = 1).long()
                y_pred = y_pred + pred.cpu().detach().numpy().flatten().tolist()
                y = batch.y.long()
                y_label = y_label + y.cpu().detach().numpy().flatten().tolist()
                
            acc_test = accuracy_score(y_pred, y_label)
            
            print("Split {}: acc_test: {:.4f}".format(i, acc_test),)

            test_accs.append(acc_test)
            
        print("Final result: acc_test: {:.4f}+-{:.4f}".format(np.mean(test_accs), np.std(test_accs)))

        
    def train_GMNET(self, train_set):
        self.GMNET.cuda()
        
        train_set = TripleSet(train_set)
        train_loader = DataLoader(train_set, batch_size = self.GMNET_conf['bs'], shuffle = True, num_workers = 8)
        optimizer = optim.Adam(self.GMNET.parameters(), lr = self.GMNET_conf['lr'], weight_decay=1e-4)
        
        for epoch in range(1, self.GMNET_conf['epochs'] + 1):
            print("====epoch {} ====".format(epoch))
            self.GMNET.train()
            train_loss = 0.0
            for data_batch in train_loader:
                anchor_data, pos_data, neg_data = data_batch
                anchor_data, pos_data, neg_data = anchor_data.cuda(), pos_data.cuda(), neg_data.cuda()

                optimizer.zero_grad()
                
                x_1, y = self.GMNET(anchor_data, pos_data, pred_head = False)
                x_2, z = self.GMNET(anchor_data, neg_data, pred_head = False)
                
                loss = triplet_loss(x_1, y, x_2, z)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                train_loss += loss
                
            print("Epoch [{}] Train_loss {}".format(epoch, train_loss / len(train_loader)))
        
        print("GMNET training done.")
        self.GMNET.eval()
        
        
    def Mixup(self, batch, alpha, sim_method = 'cos', normalize_method = 'softmax', temperature = 1.0,):    
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 0.5
            
        lam = max(lam, 1 - lam)
        
        batch = batch.cuda()
        
        batch1 = batch.clone()

        data_list = list(batch1.to_data_list())
        
        import random
        data_list2 = data_list.copy()
        random.shuffle(data_list2)
        
        batch2 =  Batch.from_data_list(data_list2).cuda()

        h1, h2 = self.GMNET.dis_encoder(batch1, batch2, node_emd = True)
        h1, h2 = h1.detach(), h2.detach()
        
        for i in range(len(data_list)):
            data_list[i].emb = h1[batch1._slice_dict['x'][i] : batch1._slice_dict['x'][i + 1],:]

            data_list2[i].emb = h2[batch2._slice_dict['x'][i] : batch2._slice_dict['x'][i + 1],:]

        batch_size = len(data_list)
        
        mixed_data_list = []

        for i in range(len(data_list)):
            # match = data_list[i].emb @ data_list2[i].emb.T
            if sim_method == 'cos':
                emb1 = data_list[i].emb / data_list[i].emb.norm(dim = 1)[:,None]
                emb2 = data_list2[i].emb / data_list2[i].emb.norm(dim = 1)[:,None]
                match = emb1 @ emb2.T / temperature 
            elif sim_method == 'abs_diff':
                match = -(data_list[i].emb.unsqueeze(1) - data_list2[i].emb.unsqueeze(0)).norm(dim = -1)

            if (normalize_method == 'softmax'):
                normalized_match = softmax(match.detach().clone(), dim = 0)
            elif(normalize_method == 'sinkhorn'):
                normalized_match = Sinkhorn(match.detach().clone())
            
            mixed_adj = lam * to_dense_adj(data_list[i].edge_index)[0].double()+ (1-lam) * normalized_match.double() @ to_dense_adj(data_list2[i].edge_index)[0].double() @ normalized_match.double().T

            mixed_adj[mixed_adj < 0.1] = 0
            
            mixed_x = lam * data_list[i].x + (1-lam) * normalized_match.float() @ data_list2[i].x

            edge_index, edge_weights = dense_to_sparse(mixed_adj)
            
            data = Data(x = mixed_x.float(), edge_index = edge_index, edge_weights = edge_weights, y1 = data_list[i].y, y2 = data_list2[i].y)

            mixed_data_list.append(data)
            
        b = Batch.from_data_list(mixed_data_list)
        b.lam = lam
        return b
                    
        

