# Contributing to DIG

Thank you very much for your interest in contributing to Dive into Graphs (DIG)!

## Table of Contents

#### 0. Overview
#### 1. Environment & Setup
#### 2. Issues & Bug Reporting
#### 3. Adding New Models
#### 4. Adding Datasets
#### 5. Adding Evaluation Metrics
#### 6. Pull Requests

## 0. Overview

Before contributing, it is helpful to have an understanding of the general structure of DIG. Namely, DIG is not one single graph model; instead, it is a collection of methods across different topics. The objective of DIG is to enable researchers to benchmark their work and implement new ideas.

Structurally, DIG is divided into four topics: Graph Generation, Self-supervised Learning on Graphs, Interpretability of Graph Neural Networks, and Deep Learning on 3D Graphs.

Each topic has its own sets of standards and evaluation metrics, but a uniform format is used. Specifically, every directory under the `/dig` sources root contains several implemented methods, a directory of datasets used by those methods, and a directory with utilities or metrics to augment and evaluate the model. By principle, DIG strives to use the uniform tools whenever possible. So, for example, it is preferable that a contribution to Graph Generation is evaluated against the metrics in that topic.

## 1. Environment & Setup

DIG uses Git as its version control system. These instructions will help with synchronization of your local environment and this repository.

To facilitate this, fork a local copy of DIG by clicking "Fork" in the top right of the screen at [this URL](https://github.com/divelab/DIG).

Then, clone your fork:

```
$ git clone https://github.com/[YOUR_USERNAME]/DIG.git
```

Next, add the official DIG repository as a remote:

```
$ git remote add upstream https://github.com/divelab/DIG.git
```

It is recommended that a separate branch is used during development:

```
$ git branch [BRANCH_NAME]
$ git checkout [BRANCH_NAME]
```

Finally, once the contributions are ready to be pushed back to the original repository, execute the following:

```
git checkout master
git pull upstream master
git push origin master  # This updates your fork of the repository
git branch -D [BRANCH_NAME]  # This deletes the branch you made during development
```

#### Note about requirements:

Requirements are managed at the model level; however, there are several standards across DIG:

* Python = 3.7
* CUDA Toolkit = 10.1

## 2. Issues & Bug Reporting

We use the GitHub issues tracker to manage any issues, questions, and reports. Please use the label feature to indicate what topic your issue concerns.

## 3. Adding New Models

We welcome contributions of new models that are implemented in PyTorch. To contribute, follow these steps:

1. Create a new directory under the relevant topic. For example, `DIG/dig/ggraph/[YOUR_MODEL]`.

2. Add your model to this directory. Take care to craft a README that is compatible with the DIG standards. An example of a strong README is [available here](https://github.com/divelab/DIG/blob/main/dig/ggraph/GraphEBM/README.md).

3. Follow the steps provided in the Environment & Setup section to push your contributions to your fork. Then, navigate to your fork on GitHub, and create a pull request. The pull request will be reviewed by a member familiar with the parent topic of the contributed model.

4. Organize data within the `datasets` directory of the parent topic, and add any additional metrics to the `metrics` or `utils` of the parent topic.

5. Take note of any comments made to the pull request, and work with the reviewer to implement any changes.

6. That's it. Thank you for your contributions!
