import os
import time
import torch
import hydra
from omegaconf import OmegaConf
from benchmarks.xgraph.utils import check_dir
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader
from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.evaluation import XCollector
from torch_geometric.utils import add_self_loops
from dig.xgraph.models import GCN_3l, GIN_3l

IS_FRESH = False

random_seed = 123


@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    import random
    import numpy as np
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config.models.gnn_saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    config.explainers.explanation_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    print(OmegaConf.to_yaml(config))

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    dataset = get_dataset(config.datasets.dataset_root,
                          config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    if config.datasets.dataset_name == 'tox21':
        dataset.data.y = dataset.data.y[:, 2].squeeze().long()
    else:
        dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices[:20]

    if config.explainers.param.subgraph_building_method == 'split':
        config.models.param.add_self_loop = False

    if config.model_name == 'GCN_3l':
        model = GCN_3l(model_level='graph',
                       dim_node=dataset.num_node_features,
                       dim_hidden=300,
                       num_classes=dataset.num_classes)

    elif config.model_name == 'GIN_3l':
        model = GIN_3l(model_level='graph',
                       dim_node=dataset.num_node_features,
                       dim_hidden=300,
                       num_classes=dataset.num_classes)

    # state_dict = torch.load(os.path.join(config.models.gnn_saving_dir,
    #                                      config.datasets.dataset_name,
    #                                      f"{config.models.gnn_name}_"
    #                                      f"{len(config.models.param.gnn_latent_dim)}l_best.pth"))['net']
    # model.load_state_dict(state_dict)

    ckpt = torch.load(os.path.join(config.models.gnn_saving_dir,
                                   config.datasets.dataset_name,
                                   f"{config.model_name}",
                                   f"{config.model_name}_best.ckpt"))

    model.load_state_dict(ckpt['state_dict'])
    model.to(device)

    if config.model_name == 'GCN_3l':
        explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                              config.datasets.dataset_name,
                                              config.models.gnn_name,
                                              config.explainers.param.reward_method)

    elif config.model_name == 'GIN_3l':
        explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                              config.datasets.dataset_name,
                                              config.model_name,
                                              config.explainers.param.reward_method)

    check_dir(explanation_saving_dir)
    plot_utils = PlotUtils(dataset_name=config.datasets.dataset_name, is_show=False)

    if config.models.param.graph_classification:
        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)

        index = 0
        x_collector = XCollector()
        total_time = 0.0
        time_duration = 0.0
        for i, data in enumerate(dataset[test_indices]):
            index += 1
            data.to(device)
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

            from torch_geometric.data import Batch
            prediction = model(data=Batch.from_data_list([data])).argmax(-1).item()

            tic = time.time()
            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=prediction,
                                  saved_MCTSInfo_list=None)
            toc = time.time()

            time_duration += toc - tic
            total_time += time_duration

            explain_result = [explain_result]
            related_preds = [related_preds]
            x_collector.collect_data(explain_result, related_preds, label=0)
            print(f"The time duration is {time_duration}.")

    # for node level explanation
    else:
        x_collector = XCollector()
        data = dataset.data
        data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        predictions = model(data).argmax(-1)

        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)

        for node_idx in node_indices:
            data.to(device)
            saved_MCTSInfo_list = None

            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')) and not IS_FRESH:
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir,
                                                              f'example_{node_idx}.pt'))
                print(f"load example {node_idx}.")

            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  node_idx=node_idx,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=predictions[node_idx].item(),
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)

            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

            title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
                             f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
                             f'spar: {related_preds["sparsity"]:.3f}'

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            subgraphx.visualization(explain_result,
                                    y=data.y,
                                    max_nodes=config.explainers.max_ex_size,
                                    plot_utils=plot_utils,
                                    title_sentence=title_sentence,
                                    vis_name=os.path.join(explanation_saving_dir,
                                                          f'example_{node_idx}.png'))

            explain_result = [explain_result]
            related_preds = [related_preds]
            x_collector.collect_data(explain_result, related_preds, label=0)

    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

    print(f"{total_time  / len(test_indices)}")


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=subgraphx')
    pipeline()
