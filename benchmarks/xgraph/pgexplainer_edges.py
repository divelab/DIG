import os
import torch
import hydra
from omegaconf import OmegaConf
from dig.xgraph.method import PGExplainer
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.base_explainer import ExplainerBase
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader

from torch import Tensor
from typing import List, Dict, Tuple
from torch_geometric.utils import add_self_loops


class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(model=model,
                         explain_graph=pgexplainer.explain_graph,
                         molecule=molecule)
        self.explainer = pgexplainer

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs)\
            -> Tuple[List, List, List[Dict]]:
        # set default subgraph with 10 edges

        pred_label = kwargs.get('pred_label')
        num_classes = kwargs.get('num_classes')
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(self.device)

        if self.explain_graph:
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
                                                  tmp=1.0,
                                                  training=False)

        else:
            node_idx = kwargs.get('node_idx')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            x, edge_index, _, subset, _ = self.explainer.get_subgraph(node_idx, x, edge_index)
            self.hard_edge_mask = edge_index.new_empty(edge_index.size(1),
                                                       device=self.device,
                                                       dtype=torch.bool)
            self.hard_edge_mask.fill_(True)

            new_node_idx = torch.where(subset == node_idx)[0]
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
                                                  tmp=1.0,
                                                  training=False,
                                                  node_idx=new_node_idx)

        # edge_masks
        edge_masks = [edge_mask for _ in range(num_classes)]
        # Calculate mask
        hard_edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity')).sigmoid()
                           for _ in range(num_classes)]

        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            if self.explain_graph:
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)
            else:
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks,
                                                       node_idx=new_node_idx)

        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds


@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    config.models.gnn_saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    config.explainers.explanation_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    config.explainers.explainer_saving_dir = config.models.gnn_saving_dir

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
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        train_indices = loader['train'].dataset.indices
        test_indices = loader['test'].dataset.indices
    else:
        train_indices = range(len(dataset))

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)
    eval_model = get_gnnNets(input_dim=dataset.num_node_features,
                             output_dim=dataset.num_classes,
                             model_config=config.models)

    state_dict = torch.load(os.path.join(config.models.gnn_saving_dir,
                                         config.datasets.dataset_name,
                                         f"{config.models.gnn_name}_"
                                         f"{len(config.models.param.gnn_latent_dim)}l_best.pth"))['net']
    model.load_state_dict(state_dict)
    eval_model.load_state_dict(state_dict)

    model.to(device)
    eval_model.to(device)

    if config.models.param.graph_classification:
        input_dim = config.models.param.gnn_latent_dim[-1] * 2
    else:
        input_dim = config.models.param.gnn_latent_dim[-1] * 3

    pgexplainer = PGExplainer(model,
                              in_channels=input_dim,
                              device=device,
                              explain_graph=config.models.param.graph_classification,
                              epochs=config.explainers.param.ex_epochs,
                              lr=config.explainers.param.ex_learing_rate,
                              coff_size=config.explainers.param.coff_size,
                              coff_ent=config.explainers.param.coff_ent,
                              t0=config.explainers.param.t0,
                              t1=config.explainers.param.t1)

    pgexplainer_saving_path = os.path.join(config.explainers.explainer_saving_dir,
                                           config.datasets.dataset_name,
                                           config.explainers.explainer_saving_name)

    if os.path.isfile(pgexplainer_saving_path):
        print("Load saved PGExplainer model...")
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    else:
        if config.models.param.graph_classification:
            pgexplainer.train_explanation_network(dataset[train_indices[60:]])
        else:
            pgexplainer.train_explanation_network(dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)

    index = 0
    x_collector = XCollector()
    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer,
                                          model=eval_model,
                                          molecule=True)
    pgexplainer_edges.device = pgexplainer.device

    if config.models.param.graph_classification:
        for i, data in enumerate(dataset[test_indices]):
            index += 1
            data.to(device)
            prediction = model(data).softmax(dim=-1).argmax().item()
            edge_masks, hard_edge_masks, related_preds = \
                pgexplainer_edges(data.x, data.edge_index,
                                  num_classes=dataset.num_classes,
                                  sparsity=config.explainers.sparsity)
            x_collector.collect_data(edge_masks, related_preds, label=prediction)
    else:
        data = dataset.data.to(device)
        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        predictions = model(data).softmax(dim=-1).argmax(dim=-1)
        for node_idx in node_indices:
            index += 1
            data.to(device)
            with torch.no_grad():
                edge_masks, hard_edge_masks, related_preds = \
                    pgexplainer_edges(data.x, data.edge_index,
                                      node_idx=node_idx,
                                      num_classes=dataset.num_classes,
                                      sparsity=config.explainers.sparsity,
                                      pred_label=predictions[node_idx].item())
                edge_masks = [mask.detach() for mask in edge_masks]
            x_collector.collect_data(edge_masks, related_preds, label=predictions[node_idx].item())

    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=pgexplainer')
    pipeline()
