# Datasets

Here we provide the graph datasets for GNN explanation.
More details of these datasets are described in our paper.

  
## Synthetic datasets
The synthetic datasets are can be downloaded from the 
[here](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ETK3CTHgFQVFhzIFB5piofQBv2Av-TsX_rnweymigve_hg?e=SyQjEX)

#### Node classification 
* BA-shapes 
* BA-Community
* Tree-Cycle 
* Tree-Grids

#### Graph classification 
* BA-2Motifs 
* BA-LRP 

## Text2Graph datasets

Here we provide graph data for text.
For each sentence, we take tokens as nodes and dependency relationships as edges to construct graph data.
In this way, we transfer SST2, SST5 and Twitter datasets into graph classification datasets.

The corresponding Graph-SST2, Graph-SST5 and Graph-Twitter can be downloaded
[here](https://drive.google.com/drive/folders/1aWKyqXTuiWgW7vFoxa8twCYscnyT_MzO?usp=sharing).

* Graph-SST2
* Graph-SST5
* Graph-Twitter 


### Molecule datasets

Molecule datasets are shown as below. 
We directly use the molecule datasets from the [Moleculenet](http://moleculenet.ai/datasets-1)
in Pytorch-Geometric. 
These datasets will be downloaded automatically. 
 
* BBBP
* Tox21
* BACE
* ClinTox
