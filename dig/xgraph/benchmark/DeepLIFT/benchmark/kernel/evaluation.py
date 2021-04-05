"""
FileName: evaluation.py
Description: Evaluation tools
Time: 2020/7/30 11:29
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
from benchmark import val_args, test_args, data_args
from torch_geometric.data.batch import Batch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from benchmark import logger
from benchmark.kernel.utils import Metric
from sklearn.metrics import accuracy_score
from .utils import nan2zero_get_mask
import numpy as np
from tqdm import tqdm



def get_weight(stat_weight, batch_weight):
    return stat_weight / (stat_weight + batch_weight), batch_weight / (stat_weight + batch_weight)


def eval_data_preprocess(y, model_pred):
    pred = []
    target = []
    pred_prob = model_pred.softmax(dim=1)

    for i in range(pred_prob.shape[0]):
        if Metric.cur_task == 'bcs':
            if not torch.isnan(y[i]):
                pred.append(pred_prob[i, 1].item())
                target.append(y[i].item())
        elif Metric.cur_task == 'mcs':
            if not torch.isnan(y[i]):
                pred.append(pred_prob[i].tolist())
                target.append(y[i].item())
    return pred, target

def test(model: torch.nn.Module, loader: DataLoader):
    with torch.no_grad():
        model.eval()

        stat = {'score': 0, 'loss': 0, 'weight': 0}
        pred_all = []
        target_all = []
        for data in loader:
            data: Batch = data.to(val_args.device)
            ori_weight, new_weight = get_weight(stat['weight'], data.num_graphs)
            stat['weight'] = stat['weight'] + data.num_graphs

            raw_preds = model(data=data)

            # score data preparation --------------------------------------
            pred, target = eval_data_preprocess(data.y, raw_preds)
            pred_all += pred
            target_all += target

            # loss calculate ----------------------------------------

            mask, targets = nan2zero_get_mask(data, val_args)

            loss: torch.tensor = Metric.loss_func(raw_preds, targets, reduction='none') * mask
            loss = loss.sum() / mask.sum()
            stat['loss'] = stat['loss'] * ori_weight + loss.item() * new_weight

            print(f'#D#batch Loss: {loss.item():.4f}')

        # ROC_AUC score calculate ---------------------------------
        try:
            stat['score'] = np.nanmean(Metric.score_func(target_all, pred_all))
        except ValueError:
            print('#ERR#Skip score evaluation in test process due to only one class presented in y_true.')
            exit(1)

        print(f'#IN#\n-----------------------------------\n'
              f'Test {Metric.score_name}: {stat["score"]:.4f}\n'
              f'Test Loss: {stat["loss"]:.4f}')

        model.train()

    return {'score': stat['score'], 'loss': stat['loss']}


def acc_score(model: torch.nn.Module, loader: DataLoader) -> float:
    with torch.no_grad():
        model.eval()

        pred_all = []
        target_all = []
        for data in tqdm(loader):

            data: Batch = data.to(test_args.device)

            raw_preds: torch.Tensor = model(data=data)

            if 'cs' in Metric.cur_task:
                pred_labels = torch.argmax(raw_preds.softmax(1), dim=1, keepdim=True)
                pred_labels = torch.cat([pred_labels] * 2, dim=1)

            # score data preparation --------------------------------------
            pred, target = eval_data_preprocess(data.y, pred_labels)
            pred_all += pred
            target_all += target



        # Accuracy score calculate ---------------------------------
        try:
            score = accuracy_score(np.array(target_all), np.array(pred_all).round())
        except ValueError as e:
            print(f'#ERR#{e}')
            exit(1)


        print(f'#I#\n-----------------------------------\n'
              f'Test accuracy score: {score:.4f}')

        model.train()

    return score


