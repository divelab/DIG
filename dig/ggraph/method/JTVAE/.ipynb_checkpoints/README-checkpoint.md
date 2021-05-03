# JT-VAE

This is an implementation of [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364)

## Table of Contents

#### 1. Environment & Setup
#### 2. Preprocessing
#### 3. Training
#### 4. Evaluation
#### 5. Citation & Acknowledgement

## 1. Environment & Setup

#### Requirements

* Python = 3.7
* CUDA Toolkit = 10.1
* RDKit >= 2020.09.4
* PyTorch >= 1.4.0
* SciPy = 1.6.0

Use the following command to create the environment `jtvae37` with the above requirements:

```shell script
$ bash setup.sh
```

## 2. Preprocessing

### i. Building the Vocabulary
The first step of preprocessing is to build the vocabulary, which is the set of clusters that can be used in the structure by structure generation.

The available datasets are available [here](https://github.com/divelab/DIG/tree/main/dig/ggraph/datasets), and any CSV with SMILES strings can be added. Our example will use [MOSES](https://arxiv.org/abs/1811.12823).

```shell script
$ python mol_tree.py --data_file "moses.csv"
```

**NOTE**: The data file path is in reference to the [ggraph/datasets](https://github.com/divelab/DIG/tree/main/dig/ggraph/datasets) directory.

### ii. Subgraph Enumeration & Tree Decomposition
```shell script
$ mkdir moses-processed
$ python preprocess.py --data_file "moses.csv" --split 100 --jobs 16
```

## 3. Training

```shell script
$ mkdir vae_model
$ python vae_train.py --train moses-processed --vocab vocab.txt --save_dir vae_model/
```

## 4. Evaluation

Perform the inference step to generate the molecules.

```shell script
$ python sample.py --nsample 30000 --vocab vocab.txt --hidden 450 --model vae_model/model.iter-400000 > mol_samples.txt
```

## Citation & Acknowledgement

This implementation is based on [JT-VAE](https://arxiv.org/abs/1802.04364). We thank them for their awesome work!
