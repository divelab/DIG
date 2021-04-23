from dig.xgraph.dataset import BA_LRP
from dig.xgraph.models import GCN_3l
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import files_exist
import os.path as osp
import os
import cilog

cilog.create_logger(sub_print=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def split_dataset(dataset, dataset_split=[0.8, 0.1, 0.1]):
    dataset_len = len(dataset)
    dataset_split = [int(dataset_len * dataset_split[0]),
                     int(dataset_len * dataset_split[1]),
                     0]
    dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
    train_set, val_set, test_set = \
        random_split(dataset, dataset_split)

    return {'train': train_set, 'val': val_set, 'test': test_set}

dataset = BA_LRP('datasets')
dataset.data.x = dataset.data.x.to(torch.float32)
dataset.data.y = dataset.data.y[:, 0]
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_targets = dataset.num_classes
num_classes = 2

splitted_dataset = split_dataset(dataset)
dataloader = DataLoader(splitted_dataset['test'], batch_size=1, shuffle=False)


def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)

model = GCN_3l(model_level='graph', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
model.to(device)
check_checkpoints()
ckpt_path = osp.join('checkpoints', 'ba_lrp', 'GCN_3l', '0', 'GCN_3l_best.ckpt')
model.load_state_dict(torch.load(ckpt_path)['state_dict'])

# --- output an example ---
data = list(dataloader)[0].to(device)
out = model(data.x, data.edge_index)

# --- load explainer ---
from dig.xgraph.method import GNN_LRP
explainer = GNN_LRP(model, explain_graph=True)

# --- Set the Sparsity to 0.5 ---
sparsity = 0.5

# --- Create data collector and explanation processor ---
from dig.xgraph.evaluation import XCollector, ExplanationProcessor
x_collector = XCollector(sparsity)
# x_processor = ExplanationProcessor(model=model, device=device)

index = -1
for index, data in enumerate(dataloader):
    print(f'#IN#explain graph line {dataloader.dataset.indices[index] + 2}')
    data.to(device)

    if torch.isnan(data.y[0].squeeze()):
        continue

    walks, masks, related_preds = \
        explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)

    x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())

    # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
    # obtain the result: x_processor(data, masks, x_collector)

    if index >= 99:
        break

print(f'#I#Fidelity: {x_collector.fidelity:.4f}\n'
      f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')

print('end')
