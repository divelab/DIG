"""
The GOOD-Arxiv dataset adapted from `OGB
<https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html>`_ benchmark.
"""

import os
import os.path as osp

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip


class GOODArxiv(InMemoryDataset):
    r"""
    The GOOD-Arxiv dataset adapted from `OGB
    <https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html>`_ benchmark.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'degree' and 'time'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', transform=None, pre_transform=None,
                 generate: bool = False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = 'https://drive.google.com/file/d/1-Wq7PoHTAiLsos20bLlq_xNvrV5AHSWu/view?usp=sharing'

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        if shift == 'covariate':
            subset_pt = 1
        elif shift == 'concept':
            subset_pt = 2
        else:
            subset_pt = 0

        self.data, self.slices = torch.load(self.processed_paths[subset_pt])

    @property
    def raw_dir(self):
        return osp.join(self.root)

    def _download(self):
        if os.path.exists(osp.join(self.raw_dir, self.name)) or self.generate:
            return
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        self.download()

    def download(self):
        path = gdown.download(self.url, output=osp.join(self.raw_dir, self.name + '.zip'), fuzzy=True)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['no_shift.pt', 'covariate.pt', 'concept.pt']

    @staticmethod
    def load(dataset_root: str, domain: str, shift: str = 'no_shift', generate: bool = False):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        meta_info = Munch()
        meta_info.dataset_type = 'real'
        meta_info.model_level = 'node'

        dataset = GOODArxiv(root=dataset_root, domain=domain, shift=shift, generate=generate)
        dataset.data.x = dataset.data.x.to(torch.float32)
        meta_info.dim_node = dataset.num_node_features
        meta_info.dim_edge = dataset.num_edge_features

        meta_info.num_envs = (torch.unique(dataset.data.env_id) >= 0).sum()

        # Define networks' output shape.
        if dataset.task == 'Binary classification':
            meta_info.num_classes = dataset.data.y.shape[1]
        elif dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(dataset.data.y).shape[0]

        # --- clear buffer dataset._data_list ---
        dataset._data_list = None

        return dataset, meta_info
