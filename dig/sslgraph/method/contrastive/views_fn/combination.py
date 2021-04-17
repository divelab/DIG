import torch
import numpy as np
import random
from torch_geometric.data import Batch

class RandomView():
    
    def __init__(self, candidates):
        self.candidates = candidates
        
    def __call__(self, data):
        return self.views_fn(data)
    
    def views_fn(self, batch_data):
        data_list = batch_data.to_data_list()
        transformed_list = []
        for data in data_list:
            view_fn = random.choice(self.candidates)
            transformed = view_fn(data)
            transformed_list.append(transformed)
        
        return Batch.from_data_list(transformed_list)


class Sequential():
    
    def __init__(self, fn_sequence):
        self.fn_sequence = fn_sequence
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def views_fn(self, data):
        for fn in self.fn_sequence:
            data = fn(data)
        
        return data