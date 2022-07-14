Installation
======

Please follow the steps below to install DIG: Dive into Graphs.

.. note::
    We recommend you to create a virtual environment with `conda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_ and install DIG: Dive into Graphs in the virtual environment.
    

Install from pip
--------
The key dependencies of DIG: Dive into Graphs are PyTorch (>=1.10.0), PyTorch Geometric (>=2.0.0), and RDKit.

#. Install `PyTorch <https://pytorch.org/get-started/locally/>`_ (>=1.10.0).

    .. code-block:: none
    
            $ python -c "import torch; print(torch.__version__)"
            >>> 1.10.0
            
            
#. Install `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#>`_ (>=2.0.0).

    .. code-block:: none
    
            $ python -c "import torch_geometric; print(torch_geometric.__version__)"
            >>> 2.0.0
            
            
#. Install DIG: Dive into Graphs.

    .. code-block:: none
    
            pip install dive-into-graphs


After installation, you can check the version. You have successfully installed DIG: Dive into Graphs if no error occurs.

    .. code-block:: none
    
            $ python
            >>> from dig.version import __version__
            >>> print(__version__)
            
Install from source
--------
If you want to try the latest features that have not been released yet, you can install dig from source.

    .. code-block:: none
    
                git clone https://github.com/divelab/DIG.git
                cd DIG
                pip install .