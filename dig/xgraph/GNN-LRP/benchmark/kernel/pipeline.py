"""
FileName: pipeline.py
Description: Application kernel pipeline [Entry point]
Time: 2020/7/28 10:51
Project: GNN_benchmark
Author: Shurui Gui
"""

import os
import torch
from definitions import ROOT_DIR
import time
from benchmark.data import load_dataset, create_dataloader
from benchmark.models import load_model, config_model, load_explainer
from benchmark.kernel import test, init
from benchmark.kernel.utils import save_epoch, argus_parse
from benchmark.kernel.explain import XCollector, sample_explain
from benchmark.args import data_args
import matplotlib.pyplot as plt
from tqdm import tqdm
from cilog import fill_table
import shutil


# Parse arguments
print('#D#Parse arguments.')
args = argus_parse()

print(f'#IN#\n-----------------------------------\n    Task: {args["common"].task}\n'
      f'{time.asctime(time.localtime(time.time()))}')

# Initial
init(args['common'])

# Load dataset
print(f'#IN#Load Dataset {args["common"].dataset_name}')
dataset = load_dataset(args['common'].dataset_name)
print('#D#', dataset['train'][0])

# pack in batches
loader = create_dataloader(dataset)


# Load model
print('#IN#Loading model...')
model = load_model(args['common'].model_name)

if args['common'].task == 'test':

    # config model
    print('#D#Config model...')
    config_model(model, args['test'], 'test')

    # test the GNN model
    _ = test(model, loader['test'])

    print('#IN#Test end.')


elif args['common'].task == 'explain':

    # config model
    print('#IN#Config the model used to be explained...')
    config_model(model, args['explain'], 'explain')

    # create explainer
    print(f'#IN#Create explainer: {args["explain"].explainer}...')
    explainer = load_explainer(args['explain'].explainer, model, args['explain'])

    # begin explain
    explain_collector = XCollector(model, loader['explain'])
    print(f'#IN#Begin explain')
    ex_target_idx = 0 if args['common'].target_idx != -1 else args['common'].explain_idx

    explain_tik = time.time()
    DEBUG = args['explain'].debug
    if DEBUG:
        # For single visualization
        if data_args.model_level == 'node':
            data = list(loader['explain'])[0]
            node_idx = 300
            sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity,
                           node_idx=node_idx)
        else:
            index = 104
            print(f'#IN#explain graph line {loader["explain"].dataset.indices[index] + 2}')
            data = list(loader['explain'])[index]
            sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity)
    else:
        if data_args.model_level == 'node':
            index = -1
            for i, data in enumerate(loader['explain']):
                for j, node_idx in tqdm(enumerate(torch.where(data.mask == True)[0].tolist())):
                    index += 1
                    print(f'#IN#explain graph {i} node {node_idx}')
                    sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity, node_idx=node_idx)

                    if index >= 99:
                        break
                if index >= 99:
                    break
        else:
            for index, data in tqdm(enumerate(loader['explain'])):
                print(f'#IN#explain graph line {loader["explain"].dataset.indices[index] + 2}')
                sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity)

                if args['explain'].list_sample:
                    list_name = f'{args["explain"].model_name}_{args["explain"].dataset_name}_{args["explain"].sparsity}'
                    list_format = [list(range(100)),
                                   ['GradCAM', 'DeepLIFT', 'GNNExplainer', 'GNN_GI', 'WalkEraser', 'GNN_LRP', 'FlowExplainer']]
                    fill_table(os.path.join(ROOT_DIR, 'quantitative_results', f'{list_name}.xlsx'),
                               value=f'{explain_collector.fidelity:.4f}',
                               x=index, y=args['explain'].explainer,
                               table_format=list_format)
                    explain_collector.new()

                if index >= 99:
                    break

    explain_tok = time.time()


    print(f'#IM#Explain method {args["explain"].explainer}\'s performance on {args["common"].model_name} with {args["common"].dataset_name}:\n'
          f'Fidelity: {explain_collector.fidelity:.4f}\n'
          f'Infidelity: {explain_collector.infidelity:.4f}\n'
          f'RegularFidelity: {explain_collector.regular_fidelity:.4f}\n'
          f'RegularInfidelity: {explain_collector.regular_infidelity:.4f}\n'
          f'Sparsity: {explain_collector.sparsity:.4f}\n'
          f'Explain average time: {(explain_tok - explain_tik) / 100.0}s')
