import numpy as np
import torch

def adjust_learning_rate(optimizer, cur_iter, init_lr, warm_up_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if warm up step is 0, no warm up actually.
    if cur_iter < warm_up_step:
        lr = init_lr * (1. / warm_up_step + 1. / warm_up_step * cur_iter)  # [0.1lr, 0.2lr, 0.3lr, ..... 1lr]
    else:
        lr = init_lr
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_total_optim_dict(total_optim_dict, optim_dict):
    all_keys = list(optim_dict.keys())
    for key in all_keys:
        if key not in total_optim_dict:
            total_optim_dict[key] = [['', -100, -1], ['', -100, -1], ['', -100, -1], ['', -100, -1]]
        
        if optim_dict[key][0][1] > total_optim_dict[key][0][1]:
            total_optim_dict[key][0][0] = optim_dict[key][0][0]
            total_optim_dict[key][0][1] = optim_dict[key][0][1]
            total_optim_dict[key][0][2] = optim_dict[key][0][2]

        if optim_dict[key][1][1] > total_optim_dict[key][1][1]:
            total_optim_dict[key][1][0] = optim_dict[key][1][0]
            total_optim_dict[key][1][1] = optim_dict[key][1][1]
            total_optim_dict[key][1][2] = optim_dict[key][1][2]

        if optim_dict[key][2][1] > total_optim_dict[key][2][1]:
            total_optim_dict[key][2][0] = optim_dict[key][2][0]
            total_optim_dict[key][2][1] = optim_dict[key][2][1]
            total_optim_dict[key][2][2] = optim_dict[key][2][2]

        if optim_dict[key][3][1] > total_optim_dict[key][3][1]:
            total_optim_dict[key][3][0] = optim_dict[key][3][0]
            total_optim_dict[key][3][1] = optim_dict[key][3][1]
            total_optim_dict[key][3][2] = optim_dict[key][3][2]
    return total_optim_dict                     