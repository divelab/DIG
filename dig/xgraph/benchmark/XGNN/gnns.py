import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
import numpy as np
from pytorch_util import weights_init
from gcn import GCN
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import os


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    #print("===Normalizing adjacency matrix symmetrically===")
    adj = adj.numpy()
    N = adj.shape[0]
    adj = adj + np.eye(N)
    D = np.sum(adj, 0)
    D_hat = np.diag((D )**(-0.5))
    out = np.dot(D_hat, adj).dot(D_hat)
    out = torch.from_numpy(out)
    return out, out

class DisNets(nn.Module):
    def __init__(self, input_dim=7,initial_dim=8, latent_dim=[32, 48, 64], mlp_hidden = 32, num_class = 2): 
        print('Initializing DisNets')
        super(DisNets, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim  = input_dim


  #      self.input_mlp = nn.Linear(self.input_dim, initial_dim)

        self.gcns = nn.ModuleList()
        self.layer_num = len(latent_dim)
        self.gcns.append(GCN(input_dim, self.latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.gcns.append(GCN(self.latent_dim[i-1], self.latent_dim[i]))
        
        self.dense_dim = latent_dim[-1]
        
        self.Softmax = nn.Softmax(dim=0)

        self.h1_weights = nn.Linear(self.dense_dim, mlp_hidden)
        self.h2_weights = nn.Linear(mlp_hidden, num_class)
        self.mlp_non_linear = nn.ELU() 

        weights_init(self)
        
    def forward(self, node_feat, adj_matrix):
        logits, probs = self.embedding(node_feat, adj_matrix)
        return logits, probs
    
    def embedding(self, node_feat, n2n_sp):
        un_A, A = normalize_adj(n2n_sp)

        cur_out = node_feat
        cur_A = A.float()

   #     cur_out = self.input_mlp(cur_out)
  #      cur_out = self.input_non_linear(cur_out)


        for i in range(self.layer_num):
            cur_out = self.gcns[i](cur_A, cur_out)
        
        graph_embedding = torch.mean(cur_out, 0) ####  for social network, use mean
      #  print(graph_embedding)
        h1 = self.h1_weights(graph_embedding)
        h1 = self.mlp_non_linear(h1)
        logits = self.h2_weights(h1)
    #    logits = h1
        probs =  self.Softmax(logits)

        return logits, probs



def read_from_graph_raw(graph):
    n = graph.number_of_nodes()
    
    degrees = [val for (node, val) in graph.degree()]
    F= np.array(degrees)
  #  F = np.ones((n, 1))
  #  F[:n+1,0] = 1   #### current graph nodes n + candidates set k=1 so n+1

    E = np.zeros([n, n])
    E[:n,:n] = np.asarray(nx.to_numpy_matrix(graph))
    E[:n,:n] += np.eye(n)

    return F[:,None], E



  #  def (self): ### do not add more nodes

if __name__ == '__main__':
    gnnNets = DisNets()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gnnNets.parameters(), lr=0.0005)#, momentum=0.9, weight_decay=5e-4)
    
    gnnNets.train()
    print('start loading data====================')
    #### Get start with MUTAG dataset
    path_adj = './MUTAG/MUTAG_A.txt'
    path_labels = './MUTAG/MUTAG_graph_labels.txt'
    path_graph_indicator = './MUTAG/MUTAG_graph_indicator.txt'
    path_labels_node = './MUTAG/MUTAG_node_labels.txt'
    
    with open(path_labels_node, 'r') as f:
        nodes_all_temp = f.read().splitlines()
        nodes_all = [int(i) for i in nodes_all_temp] 
    
    adj_all = np.zeros((len(nodes_all), len(nodes_all)))
    
    with open(path_adj, 'r') as f:
        adj_list = f.read().splitlines()
    
    for item in adj_list:
        lr = item.split(', ')
        l = int(lr[0])
        r = int(lr[1])
        adj_all[l-1,r-1] = 1
     
    with open(path_graph_indicator, 'r') as f:
        graph_indicator_temp = f.read().splitlines()
        graph_indicator = [int(i) for i in graph_indicator_temp] 
        graph_indicator = np.array(graph_indicator)
    
    with open(path_labels, 'r') as f:
        graph_labels_temp = f.read().splitlines()
        graph_labels = [int(i) for i in graph_labels_temp]     
        
    data_feature = []
    data_adj = []
    labels = []
    
    
    for i in range(1, 189):
        idx = np.where(graph_indicator==i)
        graph_len = len(idx[0])
        adj = adj_all[idx[0][0]:idx[0][0]+graph_len, idx[0][0]:idx[0][0]+graph_len]
        data_adj.append(adj)
        label = graph_labels[i-1]
        labels.append(int(label==1)) ##### label=1 means has mutagenic effect on a bacterium
        feature = nodes_all[idx[0][0]:idx[0][0]+graph_len]
        nb_clss  = 7 
        targets=np.array(feature).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        data_feature.append(one_hot_feature)    

        
    print('Finish loading data====================')        
        
    best_acc = 0
    data_size= len(labels)
    print('The total num of dataset is ', data_size)
    for epoch in range(2000):
        orders = np.random.permutation(data_size)
        acc = []
        loss_list = []
        for i in range(data_size):
            optimizer.zero_grad()
            X = data_feature[orders[i]]
            A = data_adj[orders[i]]
            label = labels[orders[i]:orders[i]+1]
            label = np.array(label)
            label = torch.from_numpy(label)
         #   X, A = read_from_graph_raw(data)
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            logits, probs = gnnNets(X.float(), A.float())
         #   print(probs)
            _, prediction = torch.max(logits, 0)
            loss = criterion(logits[None,:], label.long())
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            
            acc.append(prediction.eq(label).item())
        print("Epoch:%d  |Loss: %.3f | Acc: %.3f"%(epoch, np.average(loss_list), np.average(acc)))    
        
        if(np.average(acc)>best_acc):
            print('saving....')
            state = {
                'net': gnnNets.state_dict(),
                'acc': np.average(acc),
                'epoch': epoch,
            }        
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = np.average(acc)
    print('best acc is', best_acc) #best acc is 0.9627659574468085
    
