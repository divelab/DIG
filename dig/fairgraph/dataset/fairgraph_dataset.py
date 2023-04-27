from torch_geometric.data import InMemoryDataset, download_url, Data
import torch
import numpy as np
import os
import math
import pandas as pd
import scipy.sparse as sp
import random
from graphsaint.minibatch import Minibatch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class POKEC(InMemoryDataset):
    r'''
        `Pockec <https://snap.stanford.edu/data/soc-pokec.html>` is a social network dataset.

        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./pokec`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            train_test_valid_split (:obj:`List`, optional): A List containing the train, test and validation mask percentages. (default: :obj:`[0.7,0.15,0.15]`)
    '''
    def __init__(self, root='./pokec', transform=None, pre_transform=None, pre_filter=None, train_test_valid_split=[0.7,0.15,0.15], batch_size=1000):
        self.name = "pokec"
        self.dataset = 'region_job'
        self.sens_attr = "region"
        self.predict_attr = "I_am_working_in_field"
        self.label_number = 50000
        self.sens_number = 20000
        self.seed = 20
        self.test_idx=False
        self.batch_size = batch_size

        # assert that train_test_valid_split is valid:
        assert len(train_test_valid_split)==3
        assert math.isclose(sum(train_test_valid_split),1.)

        [self.train_mask_split,self.test_mask_split,self.valid_mask_split] = train_test_valid_split
        
        super(POKEC, self).__init__(root, transform, pre_transform, pre_filter)
        self.adj,self.features,self.sens,self.idx_sens_train,self.minibatch = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ["region_job.csv","region_job_relationship.txt","region_job.embedding"]

    @property
    def processed_file_names(self):
        return ["pokec.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        # temporarlity retreiving dataset csv file path from direct path
        # ideally should be fetched from DIG storage repository
        # download_url(os.environ.get('POKEC__REGION_JOB'),self.raw_dir)
        # download_url(os.environ.get('POKEC__REGION_JOB_EMBEDDING'),self.raw_dir)
        # download_url(os.environ.get('POKEC__REGION_JOB_RELATIONSHIOP'),self.raw_dir)
        pass

    def create_minibatch(self):
        ids = np.arange(self.features.shape[0])
        role = {'tr':ids.copy(), 'va': ids.copy(), 'te':ids.copy()}
        train_params = {'sample_coverage': 500}
        train_phase = {'sampler': 'rw', 'num_root': self.batch_size, 'depth': 3, 'end':30}
        self.minibatch = Minibatch(self.adj, self.adj,role, train_params)
        self.minibatch.set_sampler(train_phase)

    def read_graph(self):
        """
        Returns .
        """
        print(f'Loading {self.dataset} dataset from {self.raw_paths[0]}')
        # raw_paths[0] will be region_job.csv
        idx_features_labels = pd.read_csv(os.path.join(os.getcwd(),self.raw_paths[0]))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(self.sens_attr)
        header.remove(self.predict_attr)


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values
        

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        # raw_paths[1] will be region_relationship.txt
        edges_unordered = np.genfromtxt(os.path.join(os.getcwd(),self.raw_paths[1]), dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        
        random.seed(self.seed)
        label_idx = np.where(labels>=0)[0]
        random.shuffle(label_idx)
        idx_train = label_idx[:min(int(0.1 * len(label_idx)),self.label_number)]
        idx_val = label_idx[int(0.1 * len(label_idx)):int(0.2 * len(label_idx))]
        if self.test_idx:
            idx_test = label_idx[self.label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.2 * len(label_idx)):]

        sens = idx_features_labels[self.sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = torch.LongTensor(list(sens_idx))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


    def feature_norm(self,features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]

        return 2*(features - min_values).div(max_values-min_values) - 1

    def process(self):
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = self.read_graph()
        features = self.feature_norm(features)

        labels[labels>1]=1
        sens[sens>0]=1

        self.features = features.cuda()
        self.labels = labels.cuda()
        self.idx_train = idx_train.cuda()
        self.idx_val = idx_val.cuda()
        self.idx_test = idx_test.cuda()
        self.sens = sens.cuda()
        self.idx_sens_train = idx_sens_train.long().cuda()

        self.adj = adj

        self.create_minibatch()
        
        torch.save((self.adj,self.features,self.sens,self.idx_sens_train,self.minibatch),self.processed_paths[0])
        print(f'Saved to {self.processed_paths[0]}')

    def describe(self):
        print("Pokec Pyg Dataset")
        print("Nodes:", len(self.data.x), "Edges:", len(self.data.edge_index[0]))