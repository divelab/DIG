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
QM9 connsists of about 130,000 equilibrium molecules with 12 regression targets: mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv. 

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
    num_spherical=7, num_radial=6, envelope_exponent=5, 
    num_before_skip=1, num_after_skip=2, num_output_layers=3 
    )
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

Finally, we train the model and print test results.

.. code-block ::

    from dig.threedgraph.method import run
    run3d = run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=20, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)

Output:

.. parsed-literal::

    =====Epoch 1 {'Train': 0.7765793317604924, 'Validation': 0.829788088798523, 'Test': 0.8256096243858337}
    =====Epoch 2 {'Train': 0.3513736664000606, 'Validation': 0.4852057099342346, 'Test': 0.4807925820350647}
    =====Epoch 3 {'Train': 0.2866970902528785, 'Validation': 0.3925183415412903, 'Test': 0.3926387131214142}
    =====Epoch 4 {'Train': 0.22851779905254582, 'Validation': 0.24750158190727234, 'Test': 0.24735336005687714}
    =====Epoch 5 {'Train': 0.1882370650733606, 'Validation': 0.24951410293579102, 'Test': 0.2485261708498001}
    =====Epoch 6 {'Train': 0.1729215220388387, 'Validation': 0.25098657608032227, 'Test': 0.2510414719581604}
    =====Epoch 7 {'Train': 0.16714775081184313, 'Validation': 0.08707749098539352, 'Test': 0.08680126070976257}
    =====Epoch 8 {'Train': 0.14790155550919, 'Validation': 0.22142618894577026, 'Test': 0.2184610813856125}
    =====Epoch 9 {'Train': 0.14341841166521518, 'Validation': 0.08781412988901138, 'Test': 0.08708161860704422}
    =====Epoch 10 {'Train': 0.13320978917723272, 'Validation': 0.09562557935714722, 'Test': 0.09474394470453262}
    =====Epoch 11 {'Train': 0.12690318433471445, 'Validation': 0.10588052868843079, 'Test': 0.10465845465660095}
    =====Epoch 12 {'Train': 0.11995585791554265, 'Validation': 0.1758851855993271, 'Test': 0.17593063414096832}
    =====Epoch 13 {'Train': 0.11456177215925686, 'Validation': 0.0706050917506218, 'Test': 0.06988751143217087}
    =====Epoch 14 {'Train': 0.10681104086179107, 'Validation': 0.057960350066423416, 'Test': 0.05731089040637016}
    =====Epoch 15 {'Train': 0.11122639218064642, 'Validation': 0.12124070525169373, 'Test': 0.11959604173898697}
    =====Epoch 16 {'Train': 0.05462940768005544, 'Validation': 0.04387890547513962, 'Test': 0.04229748249053955}
    =====Epoch 17 {'Train': 0.057066445401964525, 'Validation': 0.07098211348056793, 'Test': 0.07055927067995071}
    =====Epoch 18 {'Train': 0.05782321078129603, 'Validation': 0.04337486997246742, 'Test': 0.041813842952251434}
    =====Epoch 19 {'Train': 0.05424516140513118, 'Validation': 0.040363702923059464, 'Test': 0.03920820727944374}
    =====Epoch 20 {'Train': 0.0552880891001176, 'Validation': 0.040168143808841705, 'Test': 0.039053723216056824}

    Best validation MAE so far: 0.040168143808841705
    Test MAE when got best validation result: 0.039053723216056824




.. [1] Liu, M., Luo, Y., Wang, L., Xie, Y., Yuan, H., Gui, S., Yu, H., Xu, Z., Zhang, J., Liu, Y. and Yan, K., 2021. DIG: A Turnkey Library for Diving into Graph Deep Learning Research. arXiv preprint arXiv:2103.12608.
.. [2] Liu, Y., Wang, L., Liu, M., Zhang, X., Oztekin, B. and Ji, S., 2021. Spherical message passing for 3D graph networks. arXiv preprint arXiv:2102.05013.
.. [3] Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. and Dahl, G.E., 2017, July. Neural message passing for quantum chemistry. In International conference on machine learning (pp. 1263-1272). PMLR.
.. [4] Wu, Z., Ramsundar, B., Feinberg, E.N., Gomes, J., Geniesse, C., Pappu, A.S., Leswing, K. and Pande, V., 2018. MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), pp.513-530.
.. [5] Wang, Z., Liu, M., Luo, Y., Xu, Z., Xie, Y., Wang, L., Cai, L. and Ji, S., 2020. Advanced graph and sequence neural networks for molecular property prediction and drug discovery. arXiv preprint arXiv:2012.01981.
.. [6] Schütt, K.T., Sauceda, H.E., Kindermans, P.J., Tkatchenko, A. and Müller, K.R., 2018. Schnet–a deep learning architecture for molecules and materials. The Journal of Chemical Physics, 148(24), p.241722.
.. [7] Klicpera, J., Groß, J. and Günnemann, S., 2020. Directional message passing for molecular graphs. arXiv preprint arXiv:2003.03123.
.. [8] Ramakrishnan, R., Dral, P.O., Rupp, M. and Von Lilienfeld, O.A., 2014. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1), pp.1-7.

