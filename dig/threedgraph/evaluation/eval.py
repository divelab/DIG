import numpy as np
import torch

class ThreeDEvaluator:
    r"""
        Evaluator for the 3D datasets, including QM9, MD17.
        Metric is Mean Absolute Error.
    """
    def __init__(self):
        pass 

    def eval(self, input_dict):
        r"""Run evaluation.

        Args:
            input_dict (dict): A python dict with the following items: :obj:`y_true` and :obj:`y_pred`. 
             :obj:`y_true` and :obj:`y_pred` need to be of the same type (either numpy.ndarray or torch.Tensor) and the same shape.

        :rtype: :class:`dict` (a python dict with item :obj:`mae`)
        """
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}