# Datasets

## QM9

QM9 dataset includes geometric, energetic, electronic, and thermodynamic properties for 134k stable small organic molecules [(paper)](https://www.nature.com/articles/sdata201422).
We used the processed data in [DimeNet](https://github.com/klicperajo/dimenet/tree/master/data), you can also use [QM9 in Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9).

There are 12 properties. Here are the units for each property.

|                          | mu     | alpha  | homo | lumo | gap  | r2    | zpve | U0   | U    | H    | G    | Cv     | std. MAE |
| ------------------------ | ------ | ------ | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ------ | -------- |
| Unit in the dataset      | D      | a_0^3  | eV   | eV   | eV   | a_0^2 | eV   | eV   | eV   | eV   | eV   | cal / mol K  |    |
| Unit of the reported MAE | D      | a_0^3  | meV  | meV  | meV  | a_0^2 | meV  | meV  | meV  | meV  | meV  | cal / mol K  | %  |


## MD17

MD17 is a collection of eight molecular dynamics simulations for small organic molecules [(paper)](https://advances.sciencemag.org/content/3/5/e1603015.short).

The units for energy and force are kcal / mol and kcal / mol A.


## ECdataset and FOLDdataset

For ECdataset and FOLDdatset, please download datasets from [here](https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets) (Protein function and Scope 1.75) to a path. The set the parameter `root='path'` to load and process the data.

Usage example:
```python
# ECdataset
for split in ['Train', 'Val', 'Test']:
    print('#### Now processing {} data ####'.format(split))
    dataset = ECdataset(root='path', split=split)
    print(dataset)

# FOLDdataset
for split in ['training', 'validation', 'test_fold', 'test_superfamily', 'test_family']:
    print('#### Now processing {} data ####'.format(split))
    dataset = FOLDdataset(root='path', split=split)
    print(dataset)
```