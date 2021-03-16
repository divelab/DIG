# Datasets

Here we provide the datasets for graph explanation. 
More details of these datasets are described in our paper.

  
## Synthetic datasets
The synthetic datasets are can be downloaded from the 
[PGExplainer repository](https://github.com/flyingdoog/PGExplainer)
except the BA-LRP. 
The BA-LRP dataset can be download 
[here](https://mailustceducn-my.sharepoint.com/:u:/g/personal/agnesgsr_mail_ustc_edu_cn/EdH7QVBBghBBgmMgf0_UZSAByxkMa3AvRdH7_QwD9MUfrw?e=EN3JiS).

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
 These datasets will be downloaded automatically if you don't download the datasets. 
 

* BBBP
* Tox21
* BACE
* ClinTox
