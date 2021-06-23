
Tutorial for the subgraphx method
==================================


Load the dataset

.. code-block ::

    import torch
    import os.path as osp
    from dig.xgraph.dataset import SynGraphDataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = SynGraphDataset('./datasets', 'BA_shapes')
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes


Load the trained model

.. code-block ::

    from dig.xgraph.models import GCN_2l
    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    model.to(device)
    ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])


Prediction of example

.. code-block::

    data = dataset[0].to(device)
    node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    node_idx = node_indices[6]
    logits = model(data.x, data.edge_index)
    prediction = logits[node_idx].argmax(dim=-1)

Visualization of the example

.. code-block::

    from dig.xgraph.method.subgraphx import PlotUtils
    from dig.xgraph.method.subgraphx import MCTS
    from torch_geometric.utils import to_networkx

    subgraph_x, subgraph_edge_index, subset, edge_mask, kwargs = \
        MCTS.__subgraph__(node_idx, data.x, data.edge_index, num_hops=2)
    subgraph_y = data.y[subset].to('cpu')
    vis_graph = to_networkx(Data(x=subgraph_x, edge_index=subgraph_edge_index))
    plotutils = PlotUtils(dataset_name='ba_shapes')
    plotutils.plot(vis_graph, nodelist=[], figname=None, y=subgraph_y, node_idx=node_indice)

.. image:: imgs/subgraphx_ori_graph.png
    :width: 80%
    :align: center

SubgraphX class :class:`~dig.xgraph.method.SubgraphX`

Monte Carlo Tree Search :class:`~dig.xgraph.method.MCTS`

.. code-block::

    from dig.xgraph.method import SubgraphX
    explainer = SubgraphX(model, num_classes=4, device=device, explain_graph=False)

Visualization of the explanation results:

.. code-block::

    from dig.xgraph.method.subgraphx import find_closest_node_result
    plotutils = PlotUtils(dataset_name='ba_shapes')

    # Visualization
    max_nodes = 5
    node_idx = node_indices[6]
    print(f'explain graph node {node_idx}')
    data.to(device)
    logits = model(data.x, data.edge_index)
    prediction = logits[node_idx].argmax(-1).item()

    _, explanation_results, related_preds = \
        explainer(data.x, data.edge_index, node_idx=node_idx, max_nodes=max_nodes)
        result = find_closest_node_result(explanation_results[prediction], max_nodes=max_nodes)

        plotutils = PlotUtils(dataset_name='ba_shapes')
        explainer.visualization(explanation_results,
                                prediction,
                                max_nodes=max_nodes,
                                plot_utils=plotutils,
                                y=data.y)

.. image:: imgs/subgraphx_explanation.png
    :width: 80%
    :align: center

Show the fidelity and sparsity of the explanataion result.

.. code-block::

    max_nodes = 5
    node_idx = node_indices[20]
    _, explanation_results, related_preds = \
        explainer(data.x, data.edge_index, node_idx=node_idx, max_nodes=max_nodes)
    result = find_closest_node_result(explanation_results[prediction], max_nodes=max_nodes)
    related_preds[prediction]


Results:
fidelity: 0.1384, sparsity: 0.6429

