import os
import torch
import networkx as nx
from tqdm import tqdm
from models import GnnNets_NC
from load_dataset import get_dataset
from fornode.mcts import MCTS, reward_func
from shapley import gnn_score, GnnNets_NC2value_func
from torch_geometric.utils import to_networkx
from Configures import data_args, mcts_args, reward_args, model_args
from utils import PlotUtils, find_closest_node_result


def pipeline(subgraph_max_nodes):
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    data = dataset[0]
    node_indices = torch.where(data.test_mask * data.y != 0)[0]

    gnnNets = GnnNets_NC(input_dim, output_dim, model_args)
    checkpoint = torch.load(mcts_args.explain_model_path)
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    gnnNets.eval()
    save_dir = os.path.join('./results', f"{mcts_args.dataset_name}"
                                         f"_{model_args.model_name}"
                                         f"_{reward_args.reward_method}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    fidelity_score_list = []
    sparsity_score_list = []
    for node_idx in tqdm(node_indices):
        # find the paths and build the graph
        result_path = os.path.join(save_dir, f"node_{node_idx}_score.pt")

        # get data and prediction
        logits, prob,  _ = gnnNets(data.clone())
        _, prediction = torch.max(prob, -1)
        prediction = prediction[node_idx].item()

        # build the graph for visualization
        graph = to_networkx(data, to_undirected=True)
        node_labels = {k: int(v) for k, v in enumerate(data.y)}
        nx.set_node_attributes(graph, node_labels, 'label')

        #  searching using gnn score
        mcts_state_map = MCTS(node_idx=node_idx, ori_graph=graph,
                              X=data.x, edge_index=data.edge_index,
                              num_hops=len(model_args.latent_dim),
                              n_rollout=mcts_args.rollout,
                              min_atoms=mcts_args.min_atoms,
                              c_puct=mcts_args.c_puct,
                              expand_atoms=mcts_args.expand_atoms)
        value_func = GnnNets_NC2value_func(gnnNets,
                                           node_idx=mcts_state_map.node_idx,
                                           target_class=prediction)
        score_func = reward_func(reward_args, value_func)
        mcts_state_map.set_score_func(score_func)

        # get searching result
        if os.path.isfile(result_path):
            gnn_results = torch.load(result_path)
        else:
            gnn_results = mcts_state_map.mcts(verbose=True)
            torch.save(gnn_results, result_path)
        tree_node_x = find_closest_node_result(gnn_results, subgraph_max_nodes)

        # calculate the metrics
        original_node_list = [i for i in tree_node_x.ori_graph.nodes]
        masked_node_list = [i for i in tree_node_x.ori_graph.nodes
                            if i not in tree_node_x.coalition or i == mcts_state_map.node_idx]
        original_score = gnn_score(original_node_list, tree_node_x.data,
                                   value_func=value_func, subgraph_building_method='zero_filling')
        masked_score = gnn_score(masked_node_list, tree_node_x.data,
                                 value_func=value_func, subgraph_building_method='zero_filling')
        sparsity_score = 1 - len(tree_node_x.coalition)/tree_node_x.ori_graph.number_of_nodes()

        fidelity_score_list.append(original_score - masked_score)
        sparsity_score_list.append(sparsity_score)

        # visualization
        subgraph_node_labels = nx.get_node_attributes(tree_node_x.ori_graph, name='label')
        subgraph_node_labels = torch.tensor([v for k, v in subgraph_node_labels.items()])
        plotutils.plot(tree_node_x.ori_graph, tree_node_x.coalition, y=subgraph_node_labels,
                       node_idx=mcts_state_map.node_idx,
                       figname=os.path.join(save_dir, f"node_{node_idx}.png"))

    fidelity_scores = torch.tensor(fidelity_score_list)
    sparsity_scores = torch.tensor(sparsity_score_list)
    return fidelity_scores, sparsity_scores


if __name__ == '__main__':
    subgraph_max_nodes = 10
    fidelity_scores, sparsity_scores = pipeline(subgraph_max_nodes)
    print(f"fidelity score: {fidelity_scores.mean().item()}, sparsity score: {sparsity_scores.mean().item()}")
