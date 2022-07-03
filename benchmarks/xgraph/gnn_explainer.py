import os
import torch
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader, SynGraphDataset
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder, perturb_input

from dig.xgraph.method import GNNExplainer
from dig.xgraph.evaluation import XCollector
from dig.xgraph.utils.compatibility import compatible_state_dict
from torch_geometric.utils import add_remaining_self_loops


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

    # bbbp warning
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

    state_dict = compatible_state_dict(torch.load(os.path.join(
        config.models.gnn_saving_dir,
        config.datasets.dataset_name,
        f"{config.models.gnn_name}_"
        f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
    ))['net'])

    model.load_state_dict(state_dict)

    model.to(device)

    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          'GNNExplainer')
    check_dir(explanation_saving_dir)
    gnn_explainer = GNNExplainer(model,
                                 epochs=config.explainers.param.epochs,
                                 lr=config.explainers.param.lr,
                                 coff_size=config.explainers.param.coff_size,
                                 coff_ent=config.explainers.param.coff_ent,
                                 explain_graph=config.models.param.graph_classification)
    gnn_explainer_perturb = GNNExplainer(model,
                                 epochs=config.explainers.param.epochs,
                                 lr=config.explainers.param.lr,
                                 coff_size=config.explainers.param.coff_size,
                                 coff_ent=config.explainers.param.coff_ent,
                                 explain_graph=config.models.param.graph_classification)
    gnn_explainer.device = device

    index = 0
    x_collector = XCollector()
    if config.models.param.graph_classification:
        for i, data in tqdm(enumerate(dataset[test_indices])):
            index += 1
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            data.to(device)
            prediction = model(data).argmax(-1).item()

            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')):
                edge_masks = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
                edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]
                print(f"load example {test_indices[i]}.")
                edge_masks, hard_edge_masks, related_preds = \
                    gnn_explainer(data.x, data.edge_index,
                                  sparsity=config.explainers.sparsity,
                                  num_classes=dataset.num_classes,
                                  edge_masks=edge_masks)

            else:
                edge_masks, hard_edge_masks, related_preds = \
                    gnn_explainer(data.x, data.edge_index,
                                  sparsity=config.explainers.sparsity,
                                  num_classes=dataset.num_classes)
                edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
                torch.save(edge_masks, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))

            if isinstance(dataset, SynGraphDataset):
                motif_edge_mask = dataset.gen_motif_edge_mask(data)
                roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.detach().cpu().numpy())
                            for edge_mask in edge_masks]
                for target_label, related_pred in enumerate(related_preds):
                    related_preds[target_label]['accuracy'] = roc_aucs[target_label]
            x_collector.collect_data(hard_edge_masks, related_preds, label=prediction)

    else:
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        data.to(device)
        prediction = model(data).argmax(-1)
        for node_idx in node_indices:
            # if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')):
            #     edge_masks = torch.load(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))
            #     edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]
            #     print(f"load example {node_idx}.")
            #     edge_masks, hard_edge_masks, related_preds = \
            #         gnn_explainer(data.x, data.edge_index,
            #                       node_idx=node_idx,
            #                       sparsity=config.explainers.sparsity,
            #                       num_classes=dataset.num_classes,
            #                       edge_masks=edge_masks)
            # else:
            edge_masks, hard_edge_masks, related_preds = \
                gnn_explainer(data.x, data.edge_index,
                              node_idx=node_idx,
                              sparsity=config.explainers.sparsity,
                              num_classes=dataset.num_classes)

            edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
            torch.save(edge_masks, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

            if isinstance(dataset, SynGraphDataset):
                motif_edge_mask = dataset.gen_motif_edge_mask(data, node_idx=node_idx)
                motif_edge_mask = motif_edge_mask[gnn_explainer.hard_edge_mask].float()
                if not (motif_edge_mask == 1).all():
                    edge_masks = [edge_mask[gnn_explainer.hard_edge_mask] for edge_mask in edge_masks]
                    roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.detach().cpu().numpy())
                                for edge_mask in edge_masks]
                    for target_label, related_pred in enumerate(related_preds):
                        related_preds[target_label]['accuracy'] = roc_aucs[target_label]

                    # for the stability metric
                    all_roc_aucs = []
                    for label_idx in range(dataset.num_classes):
                        all_roc_aucs.append([])
                    perturb_input_list = perturb_input(data, gnn_explainer.hard_edge_mask, gnn_explainer.subset)
                    for perturb_data in perturb_input_list:
                        new_prediction = model(perturb_data)[gnn_explainer.new_node_idx].argmax(dim=-1)
                        perturb_walks, perturb_masks, perturb_related_preds = \
                            gnn_explainer_perturb(perturb_data.x,
                                                  perturb_data.edge_index,
                                                  node_idx=gnn_explainer.new_node_idx,
                                                  sparsity=config.explainers.sparsity,
                                                  num_classes=dataset.num_classes)

                        perturb_motif_edge_mask = torch.cat([
                            motif_edge_mask, torch.zeros(perturb_data.edge_index.shape[1] - motif_edge_mask.shape[0])], dim=0)
                        if not (perturb_motif_edge_mask == 1).all():
                            perturb_roc_aucs = [
                                roc_auc_score(perturb_motif_edge_mask.cpu().numpy(), edge_mask.cpu().numpy())
                                for edge_mask in perturb_masks]
                            for label_idx in range(dataset.num_classes):
                                all_roc_aucs[label_idx].append(
                                   related_preds[label_idx]['accuracy'] - perturb_roc_aucs[label_idx])

                    for target_label, related_pred in enumerate(related_preds):
                        related_preds[target_label]['stability'] = torch.tensor(
                            all_roc_aucs[target_label]).mean().item()

            x_collector.collect_data(edge_masks, related_preds, label=prediction[node_idx].item())

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

    recorder.append(experiment_settings=['gnn_explainer', f"{config.explainers.sparsity}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=gnn_explainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
