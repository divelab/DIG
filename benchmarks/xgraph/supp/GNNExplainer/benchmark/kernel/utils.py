"""
FileName: utils.py
Description: General Utils including Metrics, File Operation, Hyper-parameter parsing
Time: 2020/7/31 16:33
Project: GNN_benchmark
Author: Shurui Gui
"""

import torch
from benchmark import common_args, train_args, test_args, x_args, TrainArgs, data_args
from sklearn.metrics import roc_auc_score as sk_roc_auc, precision_recall_curve, auc, mean_squared_error, \
    mean_absolute_error, accuracy_score
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, l1_loss
import torch.nn.functional as F
from math import sqrt, isnan
import os
import shutil


def prc_auc_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(torch.tensor(y_true), torch.tensor(y_pred))
    return auc(y_true, y_pred)


def roc_auc_score(y_true, y_pred):
    return sk_roc_auc(torch.tensor(y_true), torch.tensor(y_pred))

def acc(y_true, y_pred):
    true = torch.tensor(y_true)
    pred_label = torch.tensor(y_pred)
    pred_label = pred_label.round() if data_args.model_level != 'node' else torch.argmax(pred_label, dim=1)
    return accuracy_score(true, pred_label)


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)


class Metric(object):
    loss_func = cross_entropy_with_logit
    score_func = roc_auc_score
    cur_task = ''
    set2task = {"ESOL": "reg-l1", "FreeSolv": "reg-l1", "Lipo": "reg-l1",
                "PCBA": "bcs", "MUV": "bcs", "HIV": "bcs",
                "BACE": "bcs", "BBPB": "bcs", "Tox21": "bcs",
                "ToxCast": "bcs", "SIDER": "bcs", "ClinTox": "bcs",
                "ba_lrp": "bcs", "ba_shape": "mcs"}
    set2task = {item[0].lower(): item[1] for item in set2task.items()}
    task2loss = {
        'bcs': cross_entropy_with_logit,
        'mcs': cross_entropy_with_logit,
        'reg-l1': l1_loss
    }
    set2score_name = {"ESOL": "rmse", "FreeSolv": "rmse", "Lipo": "rmse",
                      "PCBA": "prc-auc", "MUV": "prc-auc", "HIV": "roc-auc",
                      "BACE": "accuracy", "BBPB": "roc-auc", "Tox21": "accuracy",
                      "ToxCast": "roc-auc", "SIDER": "roc-auc", "ClinTox": "accuracy",
                      "ba_lrp": "accuracy", "ba_shape": "accuracy"}
    set2score_name = {item[0].lower(): item[1] for item in set2score_name.items()}
    score_name = ''

    score_name2score = {
        'rmse': rmse,
        'mae': mean_absolute_error,
        'prc-auc': prc_auc_score,
        'roc-auc': roc_auc_score,
        'accuracy': acc
    }
    lower_better = 1

    @classmethod
    def set_loss_func(cls, dataset_name):
        cls.cur_task = cls.set2task.get(dataset_name)
        cls.loss_func = cls.task2loss.get(cls.cur_task)
        assert cls.loss_func is not None

    @classmethod
    def set_score_func(cls, dataset_name):
        cls.score_func = cls.score_name2score.get(cls.set2score_name.get(dataset_name))
        assert cls.score_func is not None
        cls.score_name = cls.set2score_name.get(dataset_name).upper()
        if cls.score_name in ['RMSE', 'MAE']:
            cls.lower_better = -1
        else:
            cls.lower_better = 1


best_stat = {'score': float('nan'), 'loss': float('inf')}


def nan2zero_get_mask(data, args):
    if data_args.model_level == 'node':
        mask = data.mask
    else:
        mask = torch.ones(data.y.size()).to(args.device)
        mask[torch.isnan(data.y)] = 0
    targets = torch.clone(data.y)
    targets[torch.isnan(targets)] = 0

    return mask, targets


def save_epoch(model: torch.nn.Module, args: TrainArgs, epoch: int, stat: dir):
    global best_stat

    if isnan(best_stat['score']):
        best_stat['score'] = stat['score']

    ckpt = {
        'state_dict': model.state_dict(),
        'score': stat['score'],
        'loss': stat['loss'],
        'epoch': epoch
    }
    if not (Metric.lower_better * stat['score'] > Metric.lower_better * best_stat['score']
            or epoch % args.save_gap == 0):
        return

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        print(f'#W#Directory does not exists. Have built it automatically.\n'
              f'{os.path.abspath(args.ckpt_dir)}')
    saved_file = os.path.join(args.ckpt_dir, f'{args.model_name}_{epoch}.ckpt')
    torch.save(ckpt, saved_file)
    shutil.copy(saved_file, os.path.join(args.ckpt_dir, f'{args.model_name}_last.ckpt'))
    if Metric.lower_better * stat['score'] > Metric.lower_better * best_stat['score']:
        best_stat['score'] = stat['score']
        best_stat['loss'] = stat['loss']
        shutil.copy(saved_file, os.path.join(args.ckpt_dir, f'{args.model_name}_best.ckpt'))


def argus_parse() -> dir:

    return {'common': common_args, 'train': train_args, 'test': test_args, 'explain': x_args}

