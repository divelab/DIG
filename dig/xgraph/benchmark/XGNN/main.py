from gnn_explain import gnn_explain



explainer = gnn_explain(6, 30,  1, 50)  ####arguments: (max_node, max_step, target_class, max_iters)

explainer.train()





