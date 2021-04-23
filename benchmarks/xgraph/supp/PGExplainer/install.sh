#!/bin/bash
conda create -y -n xgraph python=3.8
source activate xgraph
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install scipy
CUDA="cu102"
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric
pip install cilog typed-argument-parser==1.5.4 tqdm
conda install -y -c conda-forge rdkit
