=======================================
Tutorial for Fair Graph representations
=======================================


In this tutorial, we will show how to learn fair graph representations. Specifically, we show how to use Graphair [1]_ to do sensitive attribute prediction on molecules. 


Learning Fair Graph representations 
===================================
We consider fair graph representation learning via data augmentations. While this direction has been explored previously, existing methods invariably rely on certain assumptions on the properties of fair graph data in order to design fixed strategies on data augmentations. Nevertheless, the exact properties of fair graph data may vary significantly in different scenarios. Hence, heuristically designed augmentations may not always generate fair graph data in different application scenarios. In this work, we propose a method, known as Graphair, to learn fair representations based on automated graph data augmentations. Such fairness-aware augmentations are themselves learned from data. Our Graphair is designed to automatically discover fairness-aware augmentations from input graphs in order to circumvent sensitive information while preserving other useful information. Experimental results demonstrate that our Graphair consistently outperforms many baselines on multiple node classification datasets in terms of fairness-accuracy trade-off performance. In addition, results indicate that Graphair can automatically learn to generate fair graph data without prior knowledge on fairness-relevant graph properties.


Code Example
================ 
We use both POKEC and NBA dataset as an example to show how to use Graphair method in our DIG library.

Firstly, we load NBA data.

.. code-block ::
    
    from dig.fairgraph.dataset import NBA
    nba = NBA()

Next, we specify the device that torch will be using.

.. code-block ::
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Next, we load Graphair model and train-evaluation function.

.. code-block ::

    from dig.fairgraph.method import run
    run_fairgraph = run()

Finally, we train the model and print test results.

.. code-block ::

    run_fairgraph.run(device,dataset=nba,model='Graphair',epochs=2_000,test_epochs=1_000,batch_size=100,
            lr=1e-4,weight_decay=1e-5)

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




.. [1] Hongyi Ling and Zhimeng Jiang and Youzhi Luo and Shuiwang Ji and Na Zou, 2023. Learning Fair Graph Representations via Automated Data Augmentations.