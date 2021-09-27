"""
FileName: model_manager.py
Description: The Controller for all Graph Neural Network models
Time: 2020/7/30 8:56
Project: GNN_benchmark
Author: Shurui Gui
"""
from inspect import isclass
import dig.xgraph.models.models as models
import torch
import sys
import os


def load_model(name) -> torch.nn.Module:
    classes = [x for x in dir(models) if isclass(getattr(models, x))]
    try:
        assert name in classes
    except:
        print('#E#Model of given name does not exist.')
        sys.exit(0)

    model = getattr(models, name)()
    print(f'#IN#{model}')

    return model


def config_model(model: torch.nn.Module, args, mode: str) -> None:
    model.to(args.device)
    model.train()

    # load checkpoint
    if mode == 'train' and args.tr_ctn:
        ckpt = torch.load(os.path.join(args.ckpt_dir, f'{args.model_name}_last.ckpt'))
        model.load_state_dict(ckpt['state_dict'])
        args.ctn_epoch = ckpt['epoch'] + 1
        print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

    if mode == 'test' or mode == 'explain':
        try:
            ckpt = torch.load(args.test_ckpt)
        except FileNotFoundError:
            print(f'#E#Checkpoint not found at {os.path.abspath(args.test_ckpt)}')
            exit(1)
        model.load_state_dict(ckpt['state_dict'])
        print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')

