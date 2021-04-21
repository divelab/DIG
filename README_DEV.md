# README for developers: instructions

---
## Table of Contents
1. [Requirements](#requirements)
1. [Installation](#installation)
1. [Add Dependencies](#add-dependencies)
1. [Architecture](#architecture)
1. [File Management](#file-management)
1. [API](#api)
1. [Writing Documentation](#writing-documentation)
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
$ source ./install
```
**Note**: CUDA is 10.1 in default. You need to modify the `install` file based on your CUDA version.

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

Generally, there are 3 directories in each part.
* `dataset` includes all datasets we need to use. A dataset is written in a class with 
  auto-downloading, auto-processing.
* `method` includes methods which share a unified input/output format.
* `evaluation` includes all the metrics to evaluate the method's outputs.

All the datasets/methods/metrics should be imported by the corresponding `__init__.py`.

`__init__.pyi` is encouraged if the APIs are sophisticated.


**Note**: Besides the dig folder (the source code of the dig package), we have another folder outside of dig, named `benchmark`. In `benchmark`, we include the reproducible code for all algorithms. It is supposed to import the reorganized code above to implement the benchmark code.


## File Management

Checkpoints/Datasets should be designed to be downloaded by the starting of the 
model/dataset class call.

The root directory of our "file system" is the `ROOT_DIR` in `dig.version`. Please
use an absolute path based on `ROOT_DIR`.

Specific locations of auto-download checkpoints/datasets are pending to be discussed.

## API

Please follow the group leader's api design.


## Writing Documentation

Please follow the steps below to write documentations

1. Install `sphinx` and `sphinx_rtd_theme`:
```bash
$ pip install sphinx
$ pip install sphinx-rtd-theme
$ pip install git+https://github.com/Chilipp/autodocsumm.git
```

2. All the documentation source files are in `DIG/docs/source/`. Find the .rst file you want to contribute and write the documentation. The language we use is [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

3. Make your html locally.
```bash
$ cd docs
$ make html
```

4. Then, you can preview your documentation by opening `DIG/docs/biuld/html/index.html`.

5. Before committing to our DIG repo, please clean the make.
```bash
$ cd docs
$ make clean
```

6. Commit your contribution to DIG repo. Then, you can check the documentations at [https://diveintographs.readthedocs.io/en/latest/](https://diveintographs.readthedocs.io/en/latest/) in about 2 minutes. If the documentation website is not updated as expected, please contact Meng. It might fail to build due to the enviroment of readthedocs.


## Comments

* After the implementation, it is necessary to add comments for the documentation. We can consider this when we write documentations.
* Please follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/) as possible in your implementation.
>>>>>>> c272837934297e24d01f8361a9ebd9067a159c87
