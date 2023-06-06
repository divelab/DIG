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

    run_fairgraph.run(device,dataset=nba,model='Graphair',epochs=2_000,test_epochs=1_000,
            lr=1e-4,weight_decay=1e-5)

Output:

.. parsed-literal::

    Epoch: 0001 sens loss: 0.7270 contrastive loss: 7.0441 edge reconstruction loss: 0.7703 feature reconstruction loss: 0.3733
    Epoch: 0002 sens loss: 0.7262 contrastive loss: 6.9677 edge reconstruction loss: 0.7623 feature reconstruction loss: 0.3744
    Epoch: 0003 sens loss: 0.7247 contrastive loss: 7.0342 edge reconstruction loss: 0.7506 feature reconstruction loss: 0.3740
    Epoch: 0004 sens loss: 0.7235 contrastive loss: 6.9492 edge reconstruction loss: 0.7338 feature reconstruction loss: 0.3749
    Epoch: 0005 sens loss: 0.7238 contrastive loss: 6.9471 edge reconstruction loss: 0.7283 feature reconstruction loss: 0.3743
    Epoch: 0006 sens loss: 0.7231 contrastive loss: 6.9312 edge reconstruction loss: 0.7163 feature reconstruction loss: 0.3712
    Epoch: 0007 sens loss: 0.7219 contrastive loss: 6.9485 edge reconstruction loss: 0.7051 feature reconstruction loss: 0.3767
    Epoch: 0008 sens loss: 0.7214 contrastive loss: 6.9033 edge reconstruction loss: 0.6952 feature reconstruction loss: 0.3702
    Epoch: 0009 sens loss: 0.7204 contrastive loss: 6.8884 edge reconstruction loss: 0.6845 feature reconstruction loss: 0.3696
    Epoch: 0010 sens loss: 0.7194 contrastive loss: 6.9383 edge reconstruction loss: 0.6713 feature reconstruction loss: 0.3743
    Epoch: 0011 sens loss: 0.7188 contrastive loss: 6.9409 edge reconstruction loss: 0.6686 feature reconstruction loss: 0.3717
    Epoch: 0012 sens loss: 0.7186 contrastive loss: 6.9375 edge reconstruction loss: 0.6573 feature reconstruction loss: 0.3735
    Epoch: 0013 sens loss: 0.7182 contrastive loss: 6.9316 edge reconstruction loss: 0.6492 feature reconstruction loss: 0.3717
    .....




.. [1] Hongyi Ling and Zhimeng Jiang and Youzhi Luo and Shuiwang Ji and Na Zou, 2023. Learning Fair Graph Representations via Automated Data Augmentations.