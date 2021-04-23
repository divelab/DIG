import torch
import torch.nn as nn
from pytorch_util import weights_init
class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, p=0.3):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=p)
        
        weights_init(self)

    def forward(self, A, X):
    #    X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return torch.relu(X)#torch.sigmoid(X)#X#torch.relu(X)#