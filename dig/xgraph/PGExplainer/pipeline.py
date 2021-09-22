import os
import glob
import time
import torch
from tqdm import tqdm
from models import GnnNets, GnnNets_NC
from utils import PlotUtils
from pgexplainer import PGExplainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from metrics import top_k_fidelity, top_k_sparsity
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, model_args, train_args


def pipeline_GC(top_k):
    dataset = get_dataset(data_args)
    if data_args.dataset_name == 'mutag':
        data_indices = list(range(len(dataset)))
        pgexplainer_trainset = dataset
    else:
        loader = get_dataloader(dataset,
                                batch_size=train_args.batch_size,
                                random_split_flag=data_args.random_split,
                                data_split_ratio=data_args.data_split_ratio,
                                seed=data_args.seed)
        data_indices = loader['test'].dataset.indices
        pgexplainer_trainset = loader['train'].dataset

    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    checkpoint = torch.load(model_args.model_path)
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    gnnNets.eval()

    save_dir = os.path.join('./results', f"{data_args.dataset_name}_"
                                         f"{model_args.model_name}_"
                                         f"pgexplainer")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    pgexplainer = PGExplainer(gnnNets)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()

    pgexplainer.get_explanation_network(pgexplainer_trainset)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration: .4}s ")

    explain_duration = 0.0
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    fidelity_score_list = []
    sparsity_score_list = []
    for data_idx in tqdm(data_indices):
        data = dataset[data_idx]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic = time.perf_counter()

        prob = pgexplainer.eval_probs(data.x, data.edge_index)
        pred_label = prob.argmax(-1).item()

        if glob.glob(os.path.join(save_dir, f"example_{data_idx}.pt")):
            file = glob.glob(os.path.join(save_dir, f"example_{data_idx}.pt"))[0]
            edge_mask = torch.from_numpy(torch.load(file))
        else:
            edge_mask = pgexplainer.explain_edge_mask(data.x, data.edge_index)
            save_path = os.path.join(save_dir, f"example_{data_idx}.pt")
            edge_mask = edge_mask.cpu()
            torch.save(edge_mask.detach().numpy(), save_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        toc = time.perf_counter()
        explain_duration += (toc - tic)

        graph = to_networkx(data)

        fidelity_score = top_k_fidelity(data, edge_mask, top_k, gnnNets, pred_label)
        sparsity_score = top_k_sparsity(data, edge_mask, top_k)

        fidelity_score_list.append(fidelity_score)
        sparsity_score_list.append(sparsity_score)

        # visualization
        if hasattr(dataset, 'supplement'):
            words = dataset.supplement['sentence_tokens'][str(data_idx)]
            plotutils.plot_soft_edge_mask(graph, edge_mask, top_k,
                                          x=data.x,
                                          words=words,
                                          un_directed=True,
                                          figname=os.path.join(save_dir, f"example_{data_idx}.png"))
        else:
            plotutils.plot_soft_edge_mask(graph, edge_mask, top_k,
                                          x=data.x,
                                          un_directed=True,
                                          figname=os.path.join(save_dir, f"example_{data_idx}.png"))

    fidelity_scores = torch.tensor(fidelity_score_list)
    sparsity_scores = torch.tensor(sparsity_score_list)
    return fidelity_scores, sparsity_scores


def pipeline_NC(top_k):
    dataset = get_dataset(data_args)
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    data = dataset[0]
    node_indices = torch.where(data.test_mask * data.y != 0)[0].tolist()

    gnnNets = GnnNets_NC(input_dim, output_dim, model_args)
    checkpoint = torch.load(model_args.model_path)
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    gnnNets.eval()

    save_dir = os.path.join('./results', f"{data_args.dataset_name}_"
                                         f"{model_args.model_name}_"
                                         f"pgexplainer")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pgexplainer = PGExplainer(gnnNets)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()

    pgexplainer.get_explanation_network(dataset, is_graph_classification=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration}s ")

    duration = 0.0
    data = dataset[0]
    fidelity_score_list = []
    sparsity_score_list = []
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    for ori_node_idx in tqdm(node_indices):
        tic = time.perf_counter()
        if glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt")):
            file = glob.glob(os.path.join(save_dir, f"node_{ori_node_idx}.pt"))[0]
            edge_mask, x, edge_index, y, subset = torch.load(file)
            edge_mask = torch.from_numpy(edge_mask)
            node_idx = int(torch.where(subset == ori_node_idx)[0])
            pred_label = pgexplainer.get_node_prediction(node_idx, x, edge_index)
        else:
            x, edge_index, y, subset, kwargs = \
                pgexplainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
            node_idx = int(torch.where(subset == ori_node_idx)[0])

            edge_mask = pgexplainer.explain_edge_mask(x, edge_index)
            pred_label = pgexplainer.get_node_prediction(node_idx, x, edge_index)
            save_path = os.path.join(save_dir, f"node_{ori_node_idx}.pt")
            edge_mask = edge_mask.cpu()
            cache_list = [edge_mask.numpy(), x, edge_index, y, subset]
            torch.save(cache_list, save_path)

        duration += time.perf_counter() - tic
        sub_data = Data(x=x, edge_index=edge_index, y=y)

        graph = to_networkx(sub_data)

        fidelity_score = top_k_fidelity(sub_data, edge_mask, top_k, gnnNets, pred_label,
                                        node_idx=node_idx, undirected=True)
        sparsity_score = top_k_sparsity(sub_data, edge_mask, top_k, undirected=True)

        fidelity_score_list.append(fidelity_score)
        sparsity_score_list.append(sparsity_score)

        # visualization
        plotutils.plot_soft_edge_mask(graph, edge_mask, top_k,
                                      y=sub_data.y,
                                      node_idx=node_idx,
                                      un_directed=True,
                                      figname=os.path.join(save_dir, f"example_{ori_node_idx}.png"))

    fidelity_scores = torch.tensor(fidelity_score_list)
    sparsity_scores = torch.tensor(sparsity_score_list)
    return fidelity_scores, sparsity_scores


def pipeline(top_k):
    if data_args.dataset_name.lower() == 'BA_shapes'.lower():
        rets = pipeline_NC(top_k)
    else:
        rets = pipeline_GC(top_k)
    return rets


if __name__ == '__main__':
    top_k = 6
    fidelity_scores, sparsity_scores = pipeline(top_k)
    print(f"fidelity score: {fidelity_scores.mean().item():.4f}, "
          f"sparsity score: {sparsity_scores.mean().item():.4f}")
