# README for developers

---
## Table of Contents
1. [Requirements](#requirements)
1. [Installation](#installation)
1. [Add Dependencies](#add-dependencies)
1. [Architecture](#architecture)
1. [File Management](#file-management)
1. [API](#api)
1. [Comments](#comments)

## Requirements

* Anaconda
* PyTorch (>=1.6)
* CUDA (>=10.0) & CUDNN (>=7.0)
* PyTorch Geometry (==1.6)
* NVIDIA GPU


## Installation

You can either install the PyTorch and PyTorch Geometry manually or 
install a `DIG` conda environments with requirements above by following:
```bash
$ ./install
```

Then you can install this package by:
```bash
$ pip install -e .
```
in this directory.

Note that: This installation is generally only for development (edit mode). The package is installed
with a symbolic link; thus, it is fine to modify the files in this package then test 
them by `from dig import XXX` directly without reinstallation.

## Add Dependencies

If there are specific dependencies that are not included by the `install` script,
please refer to the field: `install_requires` in `setup.py`. Add your requirements
into this field.

## Architecture

Please follow the [mindset](https://mm.tt/1846452931?t=Q6eSYablxF) for the architecture. 
It may be useful to refer to `xgraph`.

Generally, there are 4 directories in each part.
* `dataset` includes all datasets we need to use. A dataset is written in a class with 
  auto-downloading, auto-processing.
* `method` includes methods which share a unified input/output format.
* `evaluation` includes all the metrics to evaluate the method's outputs.
* `benchmark` is supposed to import the reorganized code above to implement reproduction.
It is proper to import the main function of different method to the `benchmark` layer by using
  `__init__.py`. e.g.
  `from dig.xgraph.benchmark.DeepLIFT.benchmark.kernel.pipeline import main` will be transformed
  into `from dig.xgraph.benchmark.deeplift`.

All the datasets/methods/metrics should be imported by the corresponding `__init__.py`.

`__init__.pyi` is encouraged if the APIs are sophisticated.

## File Management

Checkpoints/Datasets should be designed to be downloaded by the starting of the 
model/dataset class call.

The root directory of our "file system" is the `ROOT_DIR` in `dig.version`. Please
use an absolute path based on `ROOT_DIR`.

Specific locations of auto-download checkpoints/datasets are pending to be discussed.

## API

Please follow the group leader's api design.

## Comments

After the implementation, it is necessary to add comments for the documentation.