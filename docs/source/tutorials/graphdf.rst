================================
Tutorial for Graph Generation
================================


In this tutorial, we will show how to generate graphs using our DIG library [1]_. Specifically, we show how to use GraphDF [2]_ to implement a molecular graph generator with deep generative model. 


Graph Generation
===================================
In drug discovery and chemical science, a fundamental problem is to design and synthesize novel molecules with some desirable properties (e.g. high drug-likeness). This problem remains to be very challenging, because the space of molecules is discrete and very huge.
A promising solution is to construct a graph generator which can automatically generate novel molecular graphs. Recently, many approaches are proposed for molecular graph generation, such as JT-VAE [3]_, GCPN [4]_, GraphAF [5]_, GraphEBM [6]_, and GraphDF.

To generate molecular graphs, we first need to decide what is generated to form a molecular graph. Generally, the following three graph formation methods are most widely used in existing molecular graph generation approaches:

* Tree-based method. The tree structure of a molecule is firstly generated, where the nodes of tree represent a motif or subgraph of the molecular graph, e.g., an aromatic ring. Then, for any two connected subgraphs in the tree, the binding points between them are decided, and a molecular graph is finally formed by binding all subgraphs. An example of this method is JT-VAE.
* One-shot method. The molecular graph is formed by explicitly generating its node type matrix and adjacency tensor. An example of this method is GraphEBM.
* Sequential method. The molecular graph is formed step by step, where only one node or one edge is generated in each step. Examples of this method are GCPN, GraphAF and GraphDF.

After the molecular graph formation method is determined, we can use any deep generative model (e.g. VAE [7]_, GAN [8]_, and flow [9]_) to construct a graph generator, in which latent variables are mapped to the generation targets by the model. 

Next, we will show how to use GraphDF to construct a graph generator with sequential method and flow model. Specifically, we will show code examples
of random generation and property optimization. Before running the example codes, please download the `configuration files <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphDF/config>`_ and the `trained model files <https://github.com/divelab/DIG_storage/tree/main/ggraph/GraphDF/saved_ckpts>`_.


Random Generation Example
================ 
We use ZINC250k [10]_ dataset as an example to train a random molecular graph generator with GraphDF. We first load the configuration parameter and set up the dataset loader for the ZINC250k dataset.

.. code-block ::
    
    import json
    from dig.ggraph.dataset import ZINC250k
    from torch_geometric.loader import DenseDataLoader
    conf = json.load(open('config/rand_gen_zinc250k_config_dict.json'))
    dataset = ZINC250k(one_shot=False, use_aug=True)
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)

Next, we initialize a molecular graph generator, and train the model on the ZINC250k dataset. The model is trained for 10 epochs, and the checkpoints of
each epoch is saved under the folder 'rand_gen_zinc250k'.

.. code-block ::

    from dig.ggraph.method import GraphDF
    runner = GraphDF()
    lr = 0.001
    wd = 0
    max_epochs = 10
    save_interval = 1
    save_dir = 'rand_gen_zinc250k'
    runner.train_rand_gen(loader=loader, lr=lr, wd=wd, max_epochs=max_epochs, 
        model_conf_dict=conf['model'], save_interval=save_interval, save_dir=save_dir)

When the training completes, we can use the trained generator to generate molecular graphs. The generated molecules are represented by a list of rdkit.Chem.Mol objects.

.. code-block ::

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    ckpt_path = 'rand_gen_zinc250k/rand_gen_ckpt_10.pth'
    n_mols = 100
    mols, _ = runner.run_rand_gen(model_conf_dict=conf['model'], checkpoint_path=ckpt_path, 
        n_mols=n_mols, atomic_num_list=conf['atom_list'])

Finally, we can call the evaluator to evaluate the generated molecules by the validity ratio and the uniqueness ratio.

.. code-block ::

    from dig.ggraph.evaluation import RandGenEvaluator
    evaluator = RandGenEvaluator()
    input_dict = {'mols': mols}
    print('Evaluating...')
    evaluator.eval(input_dict)


Property Optimization Example
================ 
We now show how to make use of GraphDF approach to generate molecules with high penalized logP scores. In GraphDF, such molecular property optimization is done
by searching molecules in the chemical space with reinforcement learning. Specifically, we formulate the sequential generation procedure as a Markov decision
process and use Proximal Policy Optimization algorithm (a commonly used reinforcement learning algorithm) to fine-tune a pre-trained GraphDF model. 
We make the reward dependent on the penalized logP score so as to encourage the model to generate molecules with high penalized logP score.

First, we load the configuration parameter and initialize a molecular graph generator.

.. code-block ::

    import json
    from dig.ggraph.method import GraphDF
    with open('config/prop_opt_plogp_config_dict.json') as f:
        conf = json.load(f)
    runner = GraphDF()

Next, we load the pre-trained model, and start fine-tuning. 

.. code-block ::

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    pretrain_path = 'saved_ckpts/prop_opt/pretrain_plogp.pth'
    lr = 0.0001
    wd = 0
    warm_up = 0
    max_iters = 200
    save_interval = 20
    save_dir = 'prop_opt_plogp'
    runner.train_prop_opt(lr=lr, wd=wd, max_iters=max_iters, warm_up=warm_up, 
        model_conf_dict=model_conf_dict, pretrain_path=pretrain_path, 
        save_interval=save_interval, save_dir=save_dir)

When the fine-tuning completes, we can generate molecules with high penalized logP scores with the trained model. The generated molecules are represented by a list of rdkit.Chem.Mol objects.

.. code-block ::

    from dig.ggraph.evaluation import PropOptEvaluator
    checkpoint_path = 'prop_opt_plogp/prop_opt_net_199.pth'
    n_mols = 100
    mols = runner.run_prop_opt(model_conf_dict=conf['model'], checkpoint_path=checkpoint_path, 
        n_mols=n_mols, num_min_node=conf['num_min_node'], num_max_node=conf['num_max_node'], 
        temperature=conf['temperature'], atomic_num_list=conf['atom_list'])

Finally, we can call the evaluator to find the molecules with top-3 penalized logP scores among all generated molecules.

.. code-block ::

    from dig.ggraph.evaluation import PropOptEvaluator
    evaluator = PropOptEvaluator()
    input_dict = {'mols': mols}
    print('Evaluating...')
    evaluator.eval(input_dict)



.. [1] Liu, M., Luo, Y., Wang, L., Xie, Y., Yuan, H., Gui, S., Yu, H., Xu, Z., Zhang, J., Liu, Y. and Yan, K., 2021. DIG: A Turnkey Library for Diving into Graph Deep Learning Research. arXiv preprint arXiv:2103.12608.
.. [2] Luo, Y., Yan, K., and Ji, S., 2021. GraphDF: A Discrete Flow Model for Molecular Graph Generation. In the 38th International Conference on Machine Learning, 2021.
.. [3] Jin, W., Barzilay, R., and Jaakkola T., 2018. Junction Tree Variational Autoencoder for Molecular Graph Generation. In the 35th International Conference on Machine Learning, 2018.
.. [4] You, J., Liu, B., Ying, R., Pande, V., and Leskovec, J., 2018. Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation. In the 32nd Conference on Neural Information Processing Systems, 2018.
.. [5] Shi, C., Xu, M., Zhu, Z., Zhang W., Zhang M., and Tang J., 2020. GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation. In the 8th International Conference on Learning Representations, 2020.
.. [6] Liu, M., Yan, K., Oztekin, B., and Ji, S., 2021. GraphEBM: Molecular graph generation with energy-based models. arXiv preprint arXiv:2102.00546.
.. [7] Kingma, D.P., and Welling, M., 2013. Auto-encoding variational bayes. In the 2nd International Conference on Learning Representations, 2013.
.. [8] Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y., 2014. Generative adversarial networks. In the 28th Conference on Neural Information Processing Systems, 2014.
.. [9] Rezende, D., and Mohamed, S., 2015. Variational inference with normalizing flows. In the 32nd International Conference on Machine Learning, 2015.
.. [10] Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S., and Coleman, R. G., 2012. ZINC: a free tool to discover chemistry for biology. Journal of chemical information and modeling, 52(7), 1757-1768.
