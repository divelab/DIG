### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow


import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class NumpyTupleDataset(Dataset):
    """Dataset of a tuple of datasets.

        It combines multiple datasets into one dataset. Each example is represented
        by a tuple whose ``i``-th item corresponds to the i-th dataset.
        And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

        Args:
            datasets: Underlying datasets. The ``i``-th one is used for the
                ``i``-th item of each example. All datasets must have the same
                length.

        """

    def __init__(self, datasets, transform=None):
        # Load dataset
#         if not os.path.exists(filepath):
#             raise ValueError('Invalid filepath for dataset')
#         load_data = np.load(filepath)
#         datasets = []
#         i = 0
#         while True:
#             key = 'arr_{}'.format(i)
#             if key in load_data.keys():
#                 datasets.append(load_data[key]) # [(133885, 9), (133885,4,9,9), (133885, 15)]
#                 i += 1
#             else:
#                 break
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])  # 133885
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        # Initialization
        self._datasets = datasets
        self._length = length
        # self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)
        # self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, np.ndarray)):
            length = len(batches[0])
            batches = [tuple([batch[i] for batch in batches])
                    for i in range(length)]   # six.moves.range(length)]
        else:
            batches = tuple(batches)

        if self.transform:
            batches = self.transform(batches)
        return batches

    def get_datasets(self):
        return self._datasets


    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format

        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance

        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        np.savez(filepath, *numpy_tuple_dataset._datasets)
        print('Save {} done.'.format(filepath))

    @classmethod
    def load(cls, filepath, transform=None):
        print('Loading file {}'.format(filepath))
        if not os.path.exists(filepath):
            raise ValueError('Invalid filepath {} for dataset'.format(filepath))
            # return None
        load_data = np.load(filepath)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return cls(result, transform)

