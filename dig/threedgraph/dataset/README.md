# Datasets

## ECdataset and FOLDdataset

For ECdataset and FOLDdatset, please download datasets from [here](https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets) to a path. The set the parameter `root='path'` to load and process the data.

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