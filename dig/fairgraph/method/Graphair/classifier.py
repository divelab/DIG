import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(Classifier,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,h):
        return self.model(h)

    def reset_parameters(self) -> None:
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

