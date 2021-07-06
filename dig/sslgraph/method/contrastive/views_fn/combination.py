import random
from torch_geometric.data import Batch


class RandomView():
    r"""Generate views by random transformation (augmentation) on given batched graphs, 
    where each graph in the batch is treated independently. Class objects callable via 
    method :meth:`views_fn`.
    
    Args:
        candidates (list): A list of callable view generation functions (classes).
    """
    
    def __init__(self, candidates):
        self.candidates = candidates
        
    def __call__(self, data):
        return self.views_fn(data)
    
    def views_fn(self, batch_data):
        r"""Method to be called when :class:`RandomView` object is called.
        
        Args:
            batch_data (:class:`torch_geometric.data.Batch`): The input batched graphs.
            
        :rtype: :class:`torch_geometric.data.Batch`.  
        """
        data_list = batch_data.to_data_list()
        transformed_list = []
        for data in data_list:
            view_fn = random.choice(self.candidates)
            transformed = view_fn(data)
            transformed_list.append(transformed)
        
        return Batch.from_data_list(transformed_list)


class Sequential():
    r"""Generate views by applying a sequence of transformations (augmentations) on 
    given batched graphs. Class objects callable via method :meth:`views_fn`.
    
    Args:
        fn_sequence (list): A list of callable view generation functions (classes).
    """
    
    def __init__(self, fn_sequence):
        self.fn_sequence = fn_sequence
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def views_fn(self, data):
        r"""Method to be called when :class:`Sequential` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        for fn in self.fn_sequence:
            data = fn(data)
        
        return data