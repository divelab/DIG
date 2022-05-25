import os
import torch
import hydra
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import add_remaining_self_loops

from dig.xgraph.method import GradCAM
from dig.xgraph.evaluation import XCollector
from dig.xgraph.dataset import SynGraphDataset
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder


from torch_geometric.data import Data
# explain on subgraph
def perturb_input(data, hard_edge_mask, subset):
    """add 2 additional empty node into the motif graph"""
    num_add_node = 2
    num_perturb_graph = 10
    subgraph_x = data.x[subset]
    subgraph_edge_index = data.edge_index[:, hard_edge_mask]
    row, col = data.edge_index

    mappings = row.new_full((data.num_nodes,), -1)
    mappings[subset] = torch.arange(subset.size(0), device=row.device)
    subgraph_edge_index = mappings[subgraph_edge_index]

    subgraph_y = data.y[subset]

    num_node_subgraph = subgraph_x.shape[0]

    # add two nodes to the subgraph, the node features are all 0.1
    subgraph_x = torch.cat([subgraph_x,
                            torch.ones(2, subgraph_x.shape[1]).to(subgraph_x.device)],
                           dim=0)
    subgraph_y = torch.cat([subgraph_y,
                            torch.zeros(num_add_node).type(torch.long).to(subgraph_y.device)], dim=0)

    perturb_input_list = []
    for _ in range(num_perturb_graph):
        to_node = torch.randint(0, num_node_subgraph, (num_add_node,))
        frm_node = torch.arange(num_node_subgraph, num_node_subgraph + num_add_node, 1)
        add_edges = torch.cat([torch.stack([to_node, frm_node], dim=0),
                               torch.stack([frm_node, to_node], dim=0),
                               torch.stack([frm_node, frm_node], dim=0)], dim=1)
        perturb_subgraph_edge_index = torch.cat([subgraph_edge_index,
                                                 add_edges.to(subgraph_edge_index.device)], dim=1)
        perturb_input_list.append(Data(x=subgraph_x, edge_index=perturb_subgraph_edge_index, y=subgraph_y))

    return perturb_input_list


@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}.json")
    print(OmegaConf.to_yaml(config))
    fix_random_seed(config.random_seed)
    recorder = Recorder(config.record_filename)

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    dataset = get_dataset(config.datasets.dataset_root,
                          config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices
    else:
        node_indices_mask = (dataset.data.y != 0) * dataset.data.test_mask
        node_indices = torch.where(node_indices_mask)[0]

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)

    state_dict = torch.load(os.path.join(config.models.gnn_saving_dir,
                                         config.datasets.dataset_name,
                                         f"{config.models.gnn_name}_"
                                         f"{len(config.models.param.gnn_latent_dim)}l_best.pth"))['net']
    model.load_state_dict(state_dict)
    model.to(device)

    # loop for different explanation methods
    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          'GradCAM', 'Stability')
    check_dir(explanation_saving_dir)

    gc_explainer = GradCAM(model, explain_graph=config.models.param.graph_classification)
    gc_explainer_perturb = GradCAM(model, explain_graph=config.models.param.graph_classification)

    index = 0
    x_collector = XCollector()
    if config.models.param.graph_classification:
        pass
    else:
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        data.to(device)
        prediction = model(data).argmax(-1)
        for node_idx in node_indices:
            edge_masks, masks, related_preds = \
                gc_explainer(data.x, data.edge_index,
                             node_idx=node_idx,
                             sparsity=config.explainers.sparsity,
                             num_classes=dataset.num_classes)
            edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]

            if isinstance(dataset, SynGraphDataset):
                motif_edge_mask = dataset.gen_motif_edge_mask(data, node_idx=node_idx)
                motif_edge_mask = motif_edge_mask[gc_explainer.hard_edge_mask].float()
                edge_masks = [edge_mask[gc_explainer.hard_edge_mask] for edge_mask in edge_masks]
                roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.cpu().numpy())
                            for edge_mask in edge_masks]
                for target_label, related_pred in enumerate(related_preds):
                    related_preds[target_label]['accuracy'] = roc_aucs[target_label]

                # for the stability metric
                all_roc_aucs = []
                for label_idx in range(dataset.num_classes):
                    all_roc_aucs.append([])
                perturb_input_list = perturb_input(data, gc_explainer.hard_edge_mask, gc_explainer.subset)
                for perturb_data in perturb_input_list:
                    new_prediction = model(perturb_data)[gc_explainer.new_node_idx].argmax(dim=-1)
                    perturb_edge_masks, perturb_masks, perturb_related_preds = \
                        gc_explainer_perturb(perturb_data.x,
                                             perturb_data.edge_index,
                                             node_idx=gc_explainer.new_node_idx,
                                             sparsity=config.explainers.sparsity,
                                             num_classes=dataset.num_classes)
                    perturb_motif_edge_mask = torch.cat([motif_edge_mask,
                                                         torch.zeros(perturb_data.edge_index.shape[1]-motif_edge_mask.shape[0])],
                                                        dim=0)
                    perturb_roc_aucs = [roc_auc_score(perturb_motif_edge_mask.cpu().numpy(), edge_mask.cpu().numpy())
                                        for edge_mask in perturb_edge_masks]
                    for label_idx in range(dataset.num_classes):
                        all_roc_aucs[label_idx].append(related_preds[label_idx]['accuracy'] - perturb_roc_aucs[label_idx])

                for target_label, related_pred in enumerate(related_preds):
                    related_preds[target_label]['stability'] = torch.tensor(all_roc_aucs[target_label]).mean().item()

            x_collector.collect_data(masks, related_preds, label=prediction[node_idx].item())

    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv: .4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

    experiment_data = {
        'fidelity': x_collector.fidelity,
        'fidelity_inv': x_collector.fidelity_inv,
        'sparsity': x_collector.sparsity,
    }

    if x_collector.accuracy:
        print(f'Accuracy: {x_collector.accuracy}')
        experiment_data['accuracy'] = x_collector.accuracy
    if x_collector.stability:
        print(f'Stability: {x_collector.stability}')
        experiment_data['stability'] = x_collector.stability

    recorder.append(experiment_settings=['grad_cam', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=grad_cam')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()