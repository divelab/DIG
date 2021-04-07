# GraphDF

This is the official implementation for [GraphDF: A Discrete Flow Model for Molecular Graph Generation](https://arxiv.org/abs/2102.01189).

![](https://github.com/divelab/DIG/blob/main/dig/ggraph/GraphDF/figs/graphdf.png)


## Table of Contents

1. [Setup](#setup)
1. [Usage](#usage)
1. [Citation](#citation)
1. [Acknowledgement](#acknowledgement)



## Setup

To install the conda virtual environment `graphdf`:
```shell script
$ cd /ggraph/GraphDF
$ bash install.sh
```
Note that we use CUDA 10.1 in this project. If you have other CUDA versions, you should install the PyTorch and cudatoolkit compatible with your CUDA.


## Usage

### Random Generation

You can use our trained models in `GraphDF/saved_ckpts/ran_gen` or train the model from scratch:
```shell script
$ cd GraphDF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_ran_gen.py --train --out_dir=${your_model_save_dir} --data=qm9 
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_ran_gen.py --train --out_dir=${your_model_save_dir} --data=zinc250k
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_ran_gen.py --train --out_dir=${your_model_save_dir} --data=moses
```
To generate molecules using our trained model:
```shell script
$ cd GraphDF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_ran_gen.py --num_mols=100 --out_dir=${your_generation_result_save_dir} --model_dir=./saved_ckpts/ran_gen/ran_gen_qm9.pth --data=qm9
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_ran_gen.py --num_mols=100 --out_dir=${your_generation_result_save_dir} --model_dir=./saved_ckpts/ran_gen/ran_gen_zinc250k.pth --data=zinc250k
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_ran_gen.py --num_mols=100 --out_dir=${your_generation_result_save_dir} --model_dir=./saved_ckpts/ran_gen/ran_gen_moses.pth --data=moses
```

### Property Optimization

For property optimization, we aim to generate molecules with desirable properties (*i.e.*, QED and plogp in this work). You can use our trained models in `GraphDF/prop_optim` or train the model from scratch by reinforcement learning:
```shell script
$ cd GraphDF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_prop_optim.py --train --out_dir=${your_model_save_dir} --prop=plogp
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_prop_optim.py --train --out_dir=${your_model_save_dir} --prop=qed
```

To generate molecules using our trained model:
```shell script
$ cd GraphDF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_prop_optim.py --num_mols=100 --out_dir=${your_generation_result_save_dir} --model_dir=./saved_ckpts/prop_optim/prop_optim_plogp.pth --prop=plogp
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_prop_optim.py --num_mols=100 --out_dir=${your_generation_result_save_dir} --model_dir=./saved_ckpts/prop_optim/prop_optim_qed.pth --prop=qed
```

### Constrained Optimization

For constrained optimization, we aim to optimize molecules with desirable properties (plogp in this work). You can use our trained models in `GraphDF/con_optim` or train the model from scratch by reinforcement learning:
```shell script
$ cd GraphDF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_con_optim.py --train --out_dir=${your_model_save_dir} --data=graphaf
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_con_optim.py --train --out_dir=${your_model_save_dir} --data=jt
```

To optimize molecules using our trained model:
```shell script
$ cd GraphDF
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_con_optim.py --out_dir=${your_optimization_result_save_dir} --model_dir=./saved_ckpts/con_optim/con_optim_graphaf.pth --data=graphaf
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_con_optim.py --out_dir=${your_optimization_result_save_dir} --model_dir=./saved_ckpts/con_optim/con_optim_jt.pth --data=jt
```
### Citation
```
@article{luo2021graphdf,
  title={{GraphDF}: A Discrete Flow Model for Molecular Graph Generation},
  author={Luo, Youzhi and Yan, Keqiang and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2102.01189},
  year={2021}
}
```

### Acknowledgement
Our implementation is based on [GraphAF](https://github.com/DeepGraphLearning/GraphAF). Thanks a lot for their awesome works.
