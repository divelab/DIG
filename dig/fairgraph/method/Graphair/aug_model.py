import torch
import pyro

class Augmentation_model(torch.nn.Module):
    r'''
        The Augmentation model decribed in the paper `"LEARNING FAIR GRAPH REPRESENTATIONS VIA AUTOMATED DATA AUGMENTATIONS`"
    '''
    def __init__(self, temperature, x) -> None:
        super(Augmentation_model).__init__()
        self.temperature = temperature
        self.x = x # node feature matrix

    
    def forward(self):
        pass

class MLPA(torch.nn.Module):
    r'''
        The Multi layer perceptron model is used for edge perturbation.
    '''

    def __init__(self,in_features,out_features,hidden_dim=64) -> None:
        super(MLPA).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_features,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,out_features)
        )

    def normalize_adj(adj):
        adj.fill_diagonal_(1)
        # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
        D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).cuda()
        adj = D_norm @ adj @ D_norm
        return 
    
    def forward(self,encoder_out):
        Z_A = self.layer(encoder_out)
        adj_logits = Z_A @ Z_A.T
        # inner product to compute new edge probabilities
        edge_probs =torch.nn.Sigmoid(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        adj_sampled = self.normalize_adj(adj_sampled)
        return edge_probs,adj_logits

class MLPX(torch.nn.Module):
    r'''
        The Multi layer perceptron model used for node feature masking.
    '''

    def __init__(self,in_features,out_features,hidden_dim=64) -> None:
        super(MLPX).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_features,hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,out_features)
        )
    def forward(self,encoder_out):
        # encoder_out = gnn output
        # pass through MLPX
        Z_X = self.layer(encoder_out)
        node_mask_probs = torch.sigmoid(Z_X)
        mask = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=node_mask_probs).rsample()
        # compute new feature matrix with the node masking applied
        x_new = mask * self.x
        return x_new

