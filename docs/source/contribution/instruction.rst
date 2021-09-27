Contributing to DIG
=========

Thank you very much for your interest in contributing to DIG: Dive into Graphs! Any forms of contributions are welcomed, including but not limited to the following.

* `Reporting issues <#reporting-issues>`_
* `Fixing bugs <#contributing-to-code>`_
* `Adding algorithms <#contributing-to-code>`_
* `Adding datasets <#contributing-to-code>`_
* `Adding metrics <#contributing-to-code>`_
* `Adding utilities <#contributing-to-code>`_
* `Improving documentations <#improving-documentations>`_
* `Staring DIG to encourage us <https://github.com/divelab/DIG>`_

    
Overview of DIG
-------

Before contributing, it is helpful to have an understanding of the general structure of DIG. Namely, DIG is not one single graph deep learning model; instead, it is a collection of datasets, algorithms, and evaluation metrics across different topics. The objective of DIG is to provide a unified testbed for higher level, research-oriented graph deep learning tasks, which enables researchers to benchmark their work and implement new ideas.

Structurally, DIG is currently divided into four topics: Graph Generation, Self-supervised Learning on Graphs, Explainability of Graph Neural Networks, and Deep Learning on 3D Graphs.

Specifically, every directory under the :obj:`/dig` sources root contains a directory of algorithms (:obj:`method`), a directory of datasets (:obj:`dataset`), a directory of metrics (:obj:`evaluation`), and a directory of utilities (:obj:`utils`) if applicable.



Reporting issues
-------
We use the GitHub `issues <https://github.com/divelab/DIG/issues>`_ tracker to manage any issues, questions, and reports. Please use the label feature to indicate what topic your issue concerns.


Contributing to code
-------
Before you plan to fix buges, add algorithms, add datasets, add metrics, and add utilities, please firstly open an `issue <https://github.com/divelab/DIG/issues>`_ to describe the features you plan to add. We can discuss it and achieve an agreement before you start coding, which can save lots of efforts.

We explain the preferred GitHub workflow as follows.


#. Fork the DIG repository by clicking "Fork" in the top right of the screen at `this URL <https://github.com/divelab/DIG>`_.

#. Uninstall existing DIG (if applicable):

    .. code-block:: none

       pip uninstall dive-into-graphs


#. Clone your fork:

    .. code-block:: none

       git clone https://github.com/[YOUR_GITHUB_USERNAME]/DIG.git
       cd DIG

#. Install DIG in :obj:`develop` mode:

    .. code-block:: none
    
       pip install -e .
       
    .. note::
        This :obj:`develop` mode allows you to edit your code, and have the changes take effect immediately. That means you don't need to reinstall DIG after you make modifications.
    
#. Once the contributions are ready, push them to your forked repository.

#. Navigate to your fork on GitHub, and create a `pull request <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`_. The pull request will be reviewed by a member familiar with the topic.



Improving documentations
-------



#. Install sphinx, sphinx_rtd_theme, and autodocsumm:

    .. code-block:: none

        pip install sphinx
        pip install sphinx-rtd-theme
        pip install git+https://github.com/Chilipp/autodocsumm.git

#. All the documentation source files are in :obj:`DIG/docs/source/`. Find the .rst file you want to contribute and write the documentation. The language we use is `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

    .. note::
        Most documentations should be written in the code as comments, which will be converted to docs automatically.
        
#. Make your html locally.

    .. code-block:: none
    
        cd docs
        make html
        
#. Then, you can preview the documentation locally by opening :obj:`DIG/docs/biuld/html/index.html`.

#. Before pushing to the GitHub repository, please clean the make.
    .. code-block:: none
    
        cd docs
        make clean
        
#. Push the contribution to your forked repository, and then submit a pull request.