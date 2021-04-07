#!/bin/bash

conda create -n jtvae37 python=3.7
conda activate jtvae37

conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge rdkit
conda install -c conda-forge scipy=1.6.0
conda install -c conda-forge tqdm 
