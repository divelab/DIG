================================
Tutorial for 3D Graphs
================================


In this tutorial, we will show how to predict properties of 3D graphs using our DIG library [1]_. Specifically, we show how to use SphereNet [2]_ to do prediction on molecules. 


Deep Learning on 3D Graphs
===================================
Molecular property prediction is of great importance in many applications, such as chemistry, drug discovery, and material science. 
Recently, graph deep learning methods have been developed for molecular property prediction [3]_ [4]_ [5]_. 
However most of the methods only focus on 2D molecular graphs without explicitly consider 3D information, which is crucial for determining quantum chemical properties.

We consider representation learning of 3D molecular graphs in which each atom is associated with a spatial position in 3D. 
In order to yield predictions that are invariant to translation and rotation of input molecules, current methods use relative 3D information like distance [6]_, angle [7]_ and torsion [2]_ as input features.
Specifically, 
the SchNet [6]_ incorporates the distance information during the information aggregation stage by using continuous-filter convolutional layers, 
the DimeNet [7]_ explicitly considers distances between atoms and angles between edges, 
the SphereNet [2]_ uses distance, angle, and torsion for completely determining structures of 3D molecular graphs.
Generally, the use of 3D position information usually results in improved performance. 


Code Example
================ 
We use QM9 [8]_ dataset as an example to show how to use SphereNet method in our DIG library. 
QM9 consists of about 130,000 equilibrium molecules with 12 regression targets: mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv. 

Firstly, we load QM9 data. We train a separate model for each target except for gap, which was predicted by taking homo-lumo.

.. code-block ::
    
    from dig.threedgraph.dataset import QM93D
    dataset = QM93D(root='dataset/')
    target = 'U0'
    dataset.data.y = dataset.data[target]
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

Next, we load SphereNet model and evaluation function.

.. code-block ::

    from dig.threedgraph.method import SphereNet
    from dig.threedgraph.evaluation import ThreeDEvaluator
    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4, 
                      hidden_channels=128, out_channels=1, int_emb_size=64, 
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
                      num_spherical=3, num_radial=6, envelope_exponent=5, 
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

Finally, we train the model and print test results.

.. code-block ::

    from dig.threedgraph.method import run
    run3d = run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, 
              epochs=20, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)

Output:

.. parsed-literal::

    =====Epoch 1 {'Train': 0.8305539944409076, 'Validation': 0.7885677814483643, 'Test': 0.7943109273910522}
    =====Epoch 2 {'Train': 0.3417653005923415, 'Validation': 0.16290859878063202, 'Test': 0.16250823438167572}
    =====Epoch 3 {'Train': 0.2626579807482881, 'Validation': 0.10924234241247177, 'Test': 0.1091669574379921}
    =====Epoch 4 {'Train': 0.2185871605092249, 'Validation': 0.1412947177886963, 'Test': 0.14113298058509827}
    =====Epoch 5 {'Train': 0.18415136586759867, 'Validation': 0.08948442339897156, 'Test': 0.08791808038949966}
    =====Epoch 6 {'Train': 0.17059671088246983, 'Validation': 0.10857655853033066, 'Test': 0.1086759939789772}
    =====Epoch 7 {'Train': 0.15622219235277093, 'Validation': 0.08192159235477448, 'Test': 0.08170071989297867}
    =====Epoch 8 {'Train': 0.1442768630192958, 'Validation': 0.08120342344045639, 'Test': 0.08138693124055862}
    =====Epoch 9 {'Train': 0.13906806218478485, 'Validation': 0.07339970022439957, 'Test': 0.0732196718454361}
    =====Epoch 10 {'Train': 0.12617339688792625, 'Validation': 0.11456501483917236, 'Test': 0.11438193917274475}
    =====Epoch 11 {'Train': 0.12321726725571651, 'Validation': 0.0715189278125763, 'Test': 0.07092428207397461}
    =====Epoch 12 {'Train': 0.11304465457233598, 'Validation': 0.1164650246500969, 'Test': 0.11696784943342209}
    =====Epoch 13 {'Train': 0.11311055924429181, 'Validation': 0.1142609491944313, 'Test': 0.11372711509466171}
    =====Epoch 14 {'Train': 0.1103381712277869, 'Validation': 0.05894898623228073, 'Test': 0.05792950466275215}
    =====Epoch 15 {'Train': 0.09813584842398945, 'Validation': 0.13913576304912567, 'Test': 0.1383834183216095}
    =====Epoch 16 {'Train': 0.05428033658000465, 'Validation': 0.06030373275279999, 'Test': 0.059175316244363785}
    =====Epoch 17 {'Train': 0.054203004988561614, 'Validation': 0.03810606151819229, 'Test': 0.03703922778367996}
    =====Epoch 18 {'Train': 0.0530719623151666, 'Validation': 0.04359658062458038, 'Test': 0.043418560177087784}
    =====Epoch 19 {'Train': 0.05202796294149651, 'Validation': 0.04247582331299782, 'Test': 0.04204947501420975}
    =====Epoch 20 {'Train': 0.04962607438894397, 'Validation': 0.04090351238846779, 'Test': 0.040894996374845505}

    Best validation MAE so far: 0.03810606151819229
    Test MAE when got best validation result: 0.03703922778367996




.. [1] Liu, M., Luo, Y., Wang, L., Xie, Y., Yuan, H., Gui, S., Yu, H., Xu, Z., Zhang, J., Liu, Y. and Yan, K., 2021. DIG: A Turnkey Library for Diving into Graph Deep Learning Research. Journal of Machine Learning Research, 22(240), pp.1-9.
.. [2] Liu, Y., Wang, L., Liu, M., Zhang, X., Oztekin, B. and Ji, S., 2021. Spherical message passing for 3D molecular graphs. In International Conference on Learning Representations.
.. [3] Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. and Dahl, G.E., 2017, July. Neural message passing for quantum chemistry. In International conference on machine learning (pp. 1263-1272). PMLR.
.. [4] Wu, Z., Ramsundar, B., Feinberg, E.N., Gomes, J., Geniesse, C., Pappu, A.S., Leswing, K. and Pande, V., 2018. MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), pp.513-530.
.. [5] Wang, Z., Liu, M., Luo, Y., Xu, Z., Xie, Y., Wang, L., Cai, L., Qi, Q., Yuan, Z., Yang, T. and Ji, S., 2022. Advanced graph and sequence neural networks for molecular property prediction and drug discovery. Bioinformatics, 38(9), pp.2579-2586.
.. [6] Schütt, K.T., Sauceda, H.E., Kindermans, P.J., Tkatchenko, A. and Müller, K.R., 2018. Schnet–a deep learning architecture for molecules and materials. The Journal of Chemical Physics, 148(24), p.241722.
.. [7] Gasteiger, J., Groß, J. and Günnemann, S., 2019, September. Directional message passing for molecular graphs. In International Conference on Learning Representations.
.. [8] Ramakrishnan, R., Dral, P.O., Rupp, M. and Von Lilienfeld, O.A., 2014. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1), pp.1-7.

