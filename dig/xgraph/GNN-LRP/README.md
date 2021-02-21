# Brief introduction of GNN_benchmark

**Statement:** This introduction is only writen for code hacker.
It is not a **User Cookbook**!

## Environment

Please use `environment.yaml`.

`conda env create -n GNN_benchmark -f environment.yaml`

## Project structure

**Entry point**

benchmark.kernel.pipeline

*First-order Dependency*:
* benchmark.kernel.*
* benchmark.data.dataset
* benchmark.models
    * explainer_manager
    * model_manager
* benchmark
    * args
    * logger

**Nearly all of the files depend on benchmark.args and
benchmark.logger.**

**Generally, files depends on utils at the same level.**

**Models and explainers are exposed through their manager.**


## Simply start

### Train a model

**First Step**

Please check the `benchmark.data.dataset` first to know what are the supported
datasets. 

Currently, **Molecule Datasets** ("ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV",
    "BACE", "BBPB", "Tox21", "ToxCast", "SIDER", "ClinTox") are supported.
    
So that you can choose a `dataset_name` you want to use.

For multi-target dataset, if you only want to choose one target but not
all of the targets, please keep a `target_idx` that is the target index you want given
several targets.

**Second Step**

Please check the `benchmark.models.models` for supported model class names.

So you have a `model_name`.


**Third Step**

Keep your `learning rate` and `epoch`.

**Other parameters** including batch_size can also be referred in
`benchmark.args`.

Now run the following command at the root of your project in a shell. Or you can use any
IDE, but you have to tackle possible problem.

```bash
python -m benchmark.kernel.pipeline --task train --lr learning rate --model_name model_name --dataset_name dataset_name --target_idx target_idx --epoch epoch
```

An example:
```bash
python -m benchmark.kernel.pipeline --task train --lr 1e-2 --model_name GCN --dataset_name tox21 --target_idx 2 --epoch 1000
```

### Test a model

A model's checkpoint are default stored at `<filesystem> PROJECT_ROOT/checkpoints/
dataset_name/model_name/target_idx/`.

You don't have to select a checkpoint unless you insist.
GNN_benchmark will automatically load the best(highest val score) checkpoint.

We assume that you have read [Train a model](#train-a-model)

An example:
```bash
python -m benchmark.kernel.pipeline --task test --model_name GCN --dataset_name clintox --target_idx 0
```

### Explain a model

It's not completed.

Now we only support GNNExplainer for binary Graph level Classification.

If you want to have a try, just do it:

```bash
python -m benchmark.kernel.pipeline --task explain --model_name GCN --dataset_name clintox --target_idx 0
```

### what to use cilog email function?

* set your mail_setting in config/mail_setting.json
* Add a `--email` flag in the command line.

## For Developers

### How to add my own dataset?

**First**:

Add your dataset class in benchmark.data.dataset.

**Second**

Annotate benchmark.kernel.utils.Metric, then rewrite your own `Metric` class.

Method `set_loss_func` and `set_score_func` are required.

**Third**

Annotate benchmark.data.dataset.load_dataset, then rewrite your own `load_dataset` function.

**Please use `set_loss_func` and `set_score_func` in this function**

*It's better to use similar variables names of original code.
It will give me the convenience to merge your code into the benchmark.*
    
### How to add my own models?

I recommend you to use the models we GNN_benchmark have currently.

But if you need, just add a model class in benchmark.models.models.

This class name will be used as your `model_name` in the corresponding command's parameter.    
    
    
    
