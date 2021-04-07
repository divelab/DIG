# GraphEBM

This is an official implementation for [GraphEBM: Molecular Graph Generation with Energy-Based Models](https://arxiv.org/abs/2102.00546).

![](https://github.com/divelab/DIG/blob/main/dig/ggraph/GraphEBM/figs/graphebm_training.png)


## Table of Contents

1. [Setup](#setup)
1. [Usage](#usage)
1. [Citation](#citation)
1. [Acknowledgement](#acknowledgement)



## Setup

To install the conda virtual environment `graphebm`:
```shell script
$ cd /ggraph/GraphEBM
$ bash setup.sh
```
Note that we use CUDA 10.1 in this project. If you have other CUDA versions, you should install the PyTorch version compatible with your CUDA.


## Usage

### Preprocessing Data

To Convert SMILES strings to desired graphs:
```shell script
$ cd preprocess_data
$ python data_preprocess.py --data_name qm9
$ python data_preprocess.py --data_name zinc250k
```

### Random Generation

You can use our trained models in `GraphEBM/release_models` or train the model from scratch:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python train.py --save_interval=1 --save_dir=${your_model_save_dir} --max_epochs=20 --sample_step=30 --data_name=qm9 --step_size=10 --c=0.2
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python train.py --save_interval=1 --save_dir=${your_model_save_dir} --max_epochs=20 --sample_step=150 --data_name=zinc250k --step_size=30 --c=0
```
To generate molecules using the trained model:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python generate.py --model_dir=./release_models/model_qm9_uncond.pt --runs=1 --save_result_file=${your_save_result_txt_file} --sample_step=30 --data_name=qm9 --step_size=10 --c=0.2
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python generate.py --model_dir=./release_models/model_zinc250k_uncond.pt --runs=1 --save_result_file=${your_save_result_txt_file} --sample_step=150 --data_name=zinc250k --step_size=30 --c=0
```

### Goal-Directed Generation

For goal-directed generation, we aim to generate molecules with desirable properties (*i.e.*, QED and plogp in this work). You can use our trained models in `GraphEBM/release_models` or train the model from scratch:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python train_goal-directed.py --save_interval=1 --save_dir=${your_model_save_dir} --max_epochs=20  --sample_step=150 --data_name=zinc250k --step_size=30 --c=0
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python train_goal-directed.py --save_interval=1 --save_dir=${your_model_save_dir} --max_epochs=20  --sample_step=150 --data_name=zinc250k --step_size=30 --c=0 --property_name=plogp
```

To generate molecules using the trained model:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python generate_goal-directed.py --model_dir=./release_models/model_zinc250k_goal_qed.pt --runs=1 --save_result_file=${your_save_result_txt_file} --sample_step=150 --data_name=zinc250k --step_size=30 --c=0
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python generate_goal-directed.py --model_dir=./release_models/model_zinc250k_goal_plogp.pt --runs=1 --save_result_file=${your_save_result_txt_file} --sample_step=150 --data_name=zinc250k --step_size=30 --c=0 --property_name=plogp
```

### Compositional Generation

We can generate molecules with multiple objectives in a compositional manner using models trained for goal-directed generation:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python generate_compositional.py --runs=1 --save_result_file=${your_save_result_txt_file} --model_qed_dir=./release_models/model_zinc250k_goal_qed.pt --model_plogp_dir=./release_models/model_zinc250k_goal_plogp.pt --sample_step=300 --data_name=zinc250k --step_size=30 --c=0
```

### Task of Constrained Property Optimization

We can generate molecules with desired objectives(plogp) using models trained for goal-directed generation to conduct the task of constrained property optimization using 800 lowest plogp molecules in the testset:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python con_gen.py --runs=5 --sample_step=500 --exp_name=constrained_plogp_testset
```
when using 800 lowest plogp molecules in the trainset:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python con_gen_trainset.py --runs=5 --sample_step=500 --exp_name=constrained_plogp_trainset
```

### Task of Property Optimization

We can generate molecules with desired objectives(qed) using models trained for goal-directed generation to conduct the task of property optimization using the molecules in the trainset as initialization:
```shell script
$ cd GraphEBM
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python prop_optim.py --runs=1 --batch_size=10000 --exp_name=prop_optim_qed
```

### Citation
```
@article{liu2021graphebm,
      title={{GraphEBM}: Molecular Graph Generation with Energy-Based Models}, 
      author={Meng Liu and Keqiang Yan and Bora Oztekin and Shuiwang Ji},
      journal={arXiv preprint arXiv:2102.00546},
      year={2021}
}
```

### Acknowledgement
Our implementation is based on [MoFlow](https://github.com/calvin-zcx/moflow) and [IGEBM](https://github.com/rosinality/igebm-pytorch). Thanks a lot for their awesome works.
