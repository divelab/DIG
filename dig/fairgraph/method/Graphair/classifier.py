import torch
import torch.nn as nn

class Classifier(nn.Module):
    r"""
        Implementation of classifier for sensitive attribute prediction
    """
    def __init__(self, input_dim, hidden_dim) -> None:
        super(Classifier,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,h):
        return self.model(h)

