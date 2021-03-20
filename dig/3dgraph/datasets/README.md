# Datesets
## QM9
Download from https://github.com/klicperajo/dimenet/blob/master/data/qm9_eV.npz.
## MD17
Download from http://quantum-machine.org/gdml/#datasets.
Note that for Benzene dataset, we used the 2017 version.
## OC20
The dataset can be downloaded from https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md.

For the sake of simplicity and fair comparison, we recommend using the official [ocp-models](https://github.com/Open-Catalyst-Project/ocp) for training and evaluating on OC20 dataset.

In order to use our implementations, one needs to put the model file and the other used modules into their code. For example, 

- Put `3dgraph/spherenet/models.py` in `ocp/ocpmodels/models/`.
- Put `3dgraph/spherenet/features.py` and `3dgraph/utils/geometric_computing.py`  in `ocp/ocpmodels/utils/`. 

One also needs to adapt the import paths for the used modules accordingly.

Finally, the ocp code uses decorators to make the model class visible to the training module. For example, add a wrapper class like:

```python
@registry.register_model("spherenet")
class spherenetWrap(spherenet):
  ...
```

Please check other model modules in the official ocp project for details as well as how to create a config file and run the training.
