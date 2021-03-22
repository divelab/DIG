from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def compute_mae(targets, preds, num_tasks):
    mae_results = []
    for i in range(num_tasks):
        target = targets[:,i]
        pred = preds[:,i]
        mae = mean_absolute_error(target, pred)
        mae_results.append(mae)
    return mae_results

def compute_rmse(targets, preds, num_tasks):
    rmse_results = []
    for i in range(num_tasks):
        target = targets[:,i]
        pred = preds[:,i]
        rmse = np.sqrt(mean_squared_error(target, pred))
        rmse_results.append(rmse)
    return rmse_results
