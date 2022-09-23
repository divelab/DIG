import torch
import torch.nn.functional as F
import numpy as np
from torch import optim, nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from torch_geometric.data import Data

criterion = nn.BCEWithLogitsLoss(reduction = "none")
lr = 0.001
weight_decay = 0
epochs = 100


class MLP(torch.nn.Module):
    
    def __init__(self, num_layer, emb_dim, hidden_dim, out_dim=1):
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        if num_layer > 1:
            self.layers.append(nn.Linear(emb_dim, hidden_dim))
            for n in range(num_layer-1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            self.layers.append(nn.Linear(emb_dim, out_dim))
            
    def forward(self, emb):
        out = self.layers[0](emb)
        for layer in self.layers[1:]:
            out = layer(F.relu(out))
        return out
    

class EndtoEnd(torch.nn.Module):
    '''
    Class to wrap-up embedding model and downstream models into an end-to-end model.
    Args:
        embed_model, mlp_model: obj:`torch.nn.Module` objects.
        wrapped_input: Boolean. Whether (GNN) embedding model taks input wrapped in obj:`Data` object 
        or node attributes and edge indices separately.
    '''
    
    def __init__(self, embed_model, mlp_model, wrapped_input=False):
        super(EndtoEnd, self).__init__()
        self.embed_model = embed_model
        self.mlp_model = mlp_model
        self.wrapped_input = wrapped_input
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        '''
        Forward propagation outputs the final prediction.
        '''
        if self.wrapped_input:
            if batch is None:
                batch = torch.zeros_like(x[:,0], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            return self.forward_w(data)
        else:
            return self.forward_nw(x, edge_index, edge_attr, batch)
    
    def forward_w(self, data):
        self.embed_model.eval()
        with torch.no_grad():
            emb = self.embed_model(data)
        out = self.mlp_model(emb)
        return out
    
    def forward_nw(self, x, edge_index, edge_attr, batch):
        self.embed_model.eval()
        with torch.no_grad():
            emb = self.embed_model(x, edge_index, edge_attr, batch)
        out = self.mlp_model(emb)
        return out
    
    def get_emb(self, x, edge_index, edge_attr=None, batch=None):
        '''
        Forward propagation outputs only node embeddings.
        '''
        self.embed_model.eval()
        if self.wrapped_input:
            if batch is None:
                batch = torch.zeros_like(x[:,0], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            emb = self.embed_model(data)
        else:
            emb = self.embed_model(x, edge_index, edge_attr, batch)
        return emb



def train_MLP(embed_model, mlp_model, device, loader, val_loader, save_to=None):
    embed_model = embed_model.to(device)
    mlp_model = mlp_model.to(device)
    optimizer = optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_roc = 0
    for _ in range(epochs):
        print("====epoch " + str(_))
        embed_model.eval()
        mlp_model.train()
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            embeds = embed_model(batch.to(device)).detach()
            pred = mlp_model(embeds)
            y = batch.y.view(pred.shape).to(torch.float64)

            is_valid = y**2 > 0
            loss_mat = criterion(pred.double(), (y+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()

            optimizer.step()
            
        mlp_model.eval()
        y_true = []
        y_scores = []

        print("====Evaluation")
        for step, batch in enumerate(tqdm(val_loader, desc="Iteration")):

            with torch.no_grad():
                embeds = embed_model(batch.to(device)).detach()
                pred = mlp_model(embeds)

            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)

        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        
        roc_list = []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
                
        roc_score = sum(roc_list)/len(roc_list)
        print(roc_score)
        if roc_score > best_roc and save_to:
            best_roc = roc_score
            torch.save(mlp_model.state_dict(), save_to)

