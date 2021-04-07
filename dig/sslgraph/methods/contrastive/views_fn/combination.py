import torch
import numpy as np
import random
from torch_geometric.data import Batch

def random_view(candidates):
    
    def views_fn(batch_data):
        data_list = batch_data.to_data_list()
        transformed_list = []
        for data in data_list:
            view_fn = random.choice(candidates)
            transformed = view_fn(data)
            transformed_list.append(transformed)
        
        return Batch.from_data_list(transformed_list)
    
    return views_fn


def combine(fn_sequence):
    
    def views_fn(data):
        for fn in fn_sequence:
            data = fn(data)
        
        return data
    
    return views_fn