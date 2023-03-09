import torch

class Graphair(torch.nn.Module):
    r'''
        Implementation of Graphair from the paper `"LEARNING FAIR GRAPH REPRESENTATIONS VIA AUTOMATED DATA AUGMENTATIONS`"
    '''
    def __init__(self):
        super(Graphair, self).__init__()
    def forward(self,data):
        x,edge_indices = data
        # TODO: Use instances of augmentation module g, encoder f, advrsarial model k to generate all three loss components.
        pass