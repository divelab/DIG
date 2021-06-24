from dig.threedgraph.evaluation import ThreeDEvaluator
import numpy as np
import torch
import math


def test_ThreeDEvaluator():
    input_dict = {'y_true':np.array([1.0, -0.5]), 'y_pred':np.array([0.6, 0.0])}
    evaluator = ThreeDEvaluator()
    result = evaluator.eval(input_dict)
    assert len(result) == 1
    assert type(result['mae']) == float
    assert result['mae'] == 0.45

    input_dict = {'y_true':torch.Tensor([1.0, -0.5]), 'y_pred':torch.Tensor([0.6, 0.0])}
    evaluator = ThreeDEvaluator()
    result = evaluator.eval(input_dict)
    assert len(result) == 1
    assert type(result['mae']) == float
    # https://discuss.pytorch.org/t/item-gives-different-value-than-the-tensor-itself/101826
    assert torch.Tensor([result['mae']]) == torch.Tensor([0.45]) 


if __name__ == '__main__':
    test_ThreeDEvaluator()
