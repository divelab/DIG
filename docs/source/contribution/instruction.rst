Contributing to DIG
=========

Thank you very much for your interest in contributing to DIG: Dive into Graphs!

.. contents::
    :local:
    
Overview
-------

Before contributing, it is helpful to have an understanding of the general structure of DIG. Namely, DIG is not one single graph deep learning model; instead, it is a collection of datasets, algorithms, and evaluation metrics across different topics. The objective of DIG is to enable researchers to benchmark their work and implement new ideas.

Structurally, DIG is divided into four topics: Graph Generation, Self-supervised Learning on Graphs, Explainability of Graph Neural Networks, and Deep Learning on 3D Graphs.

Specifically, every directory under the `/dig` sources root contains a directory of algorithms (:obj:`method`), a directory of datasets (:obj:`dataset`), a directory of metrics (:obj:`evaluation`), and a directory of utilities (:obj:`utils`) if applicable.


Git setup
-------

#. Fork a local copy of DIG by clicking "Fork" in the top right of the screen at `this URL <(https://github.com/divelab/DIG)>`_.

#. Uninstall existing DIG (if applicable):

    .. code-block:: none

       pip uninstall dive-into-graphs


#. Clone your fork:

    .. code-block:: none

       git clone https://github.com/[YOUR_GITHUB_USERNAME]/DIG.git
       cd dig

#. Install DIG in :obj:`develop` mode:

    .. code-block:: none
    
       cd dig
       pip install -e .
       
.. note::
    This :obj:`develop` mode allows you to edit your code, and have the changes take effect immediately. That means you dont need to reinstall DIG after you make modifications.
    
#. Once the contributions are ready, pushed them to your forked repository.

#. Navigate to your fork on GitHub, and create a `pull request <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`_. The pull request will be reviewed by a member familiar with the topic.

