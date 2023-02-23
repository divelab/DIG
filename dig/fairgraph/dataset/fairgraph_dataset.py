from torch_geometric.data import InMemoryDataset, download_url, Data
import torch
import numpy as np
import os
import math
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class POKEC(InMemoryDataset):
    r'''
        `Pockec <https://snap.stanford.edu/data/soc-pokec.html>` is a social network dataset.

        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./pokec`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            train_test_valid_split (:obj:`List`, optional): A List containing the train, test and validation mask percentages. (default: :obj:`[0.7,0.15,0.15]`)
    '''
    def __init__(self, root='./pokec', transform=None, pre_transform=None, pre_filter=None, train_test_valid_split=[0.7,0.15,0.15]):
        self.name = "pokec"

        # assert that train_test_valid_split is valid:
        assert len(train_test_valid_split)==3
        assert math.isclose(sum(train_test_valid_split),1.)

        [self.train_mask_split,self.test_mask_split,self.valid_mask_split] = train_test_valid_split

        if root is None:
            root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        self.node_index_map = None
        """ Maps the original key to a node index."""

        self.node_frame = None
        """The original nodes frame."""

        self.edge_frame = None
        """The original edges frame."""
        
        super(POKEC, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[1])
    
    @property
    def raw_file_names(self):
        return ["region_job.csv","region_job_relationship.txt","region_job.embedding"]

    @property
    def processed_file_names(self):
        return ["data_frames.h5","pokec.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        # temporarlity retreiving dataset csv file path from environment var
        # ideally should be fetched from DIG storage repository
        download_url(os.environ('POKEC__REGION_JOB'),self.raw_dir)
        download_url(os.environ('POKEC__REGION_JOB_EMBEDDING'),self.raw_dir)
        download_url(os.environ('POKEC__REGION_JOB_RELATIONSHIOP'),self.raw_dir)

    def read_graph(self):
        """
        Returns Pandas node and edge dataframes.
        """
        hdf_path = os.path.join(self.processed_dir, self.processed_paths[0])
        if os.path.exists(hdf_path):
            self.node_frame = pd.read_hdf(hdf_path, "/df_node")
            self.edge_frame = pd.read_hdf(hdf_path, "/df_edge")
            return self.node_frame, self.edge_frame
        print("Reading graph from csv file.")
        # raw_paths[0] will be region_job.csv
        df_node = pd.read_csv(os.path.join(self.root,"{}.csv".format('region_job')))
        # raw_paths[1] will be region_job_relationship.txt
        df_edge = pd.read_csv(self.raw_paths[1], names = ["source", "target"], dtype=int)

        self.node_frame = df_node
        self.edge_frame = df_edge

        # save as hdf
        store = pd.HDFStore(hdf_path)
        store["dfn"] = df_node
        store["dfe"] = df_edge
        store.close()
        print("Save data frames to 'frames.h5'.")

        return df_node, df_edge

    def __transform_nodes(self):
        print("Transforming nodes")
        if self.node_frame is None:
            self.read_graph()
        # index is the user_id here
        self.node_index_map = {int(user_id): i for i, user_id in enumerate(self.node_frame.index.unique())}
        # transform nodes: remove the user_id, sensitive attribute and prediction attribute
        headers = list(self.node_frame.columns)
        headers.remove("user_id")
        headers.remove('region')
        headers.remove("I_am_working_in_field")
        x = torch.tensor(self.node_frame[headers].values, dtype = torch.float)
        return x, self.node_index_map

    def __transform_edges(self):
        print("Transforming edges")

        src = [self.node_index_map[src_id] if src_id in self.node_index_map else -1 for src_id in self.edge_frame.source]
        dst = [self.node_index_map[tgt_id] if tgt_id in self.node_index_map else -1 for tgt_id in self.edge_frame.target]
        edge_index = torch.tensor([src, dst])

        return edge_index

    def create_node_masks(self,d):
        print("Creating classification masks")
        amount = len(d.x)
        # actually the index to the nodes
        nums = np.arange(amount)
        np.random.shuffle(nums)

        train_size = int(amount * self.train_mask_split)
        test_size = int(amount * self.train_mask_split+self.test_mask_split) - train_size
        val_size = amount - train_size - test_size

        train_set = nums[0:train_size]
        test_set = nums[train_size:train_size + test_size]
        val_set = nums[train_size + test_size:]

        assert len(train_set) + len(test_set) + len(val_set) == amount, "The split should be coherent."

        train_mask = torch.zeros(amount, dtype = torch.long, device = device)
        for i in train_set:
            train_mask[i] = 1.

        test_mask = torch.zeros(amount, dtype = torch.long, device = device)
        for i in test_set:
            test_mask[i] = 1.

        val_mask = torch.zeros(amount, dtype = torch.long, device = device)
        for i in val_set:
            val_mask[i] = 1.

        d.train_mask = train_mask
        d.test_mask = test_mask
        d.val_mask = val_mask

    def process(self):
        self.read_graph()
        nodes_x, nodes_mapping = self.__transform_nodes()
        edges_index = self.__transform_edges()

        d = Data(x = nodes_x, edge_index = edges_index, edge_attr = None, y = None)

        if self.pre_filter is not None:
            d = self.pre_filter(d)

        if self.pre_transform is not None:
            d = self.pre_transform(d)
        POKEC.create_node_masks(d)
        print("Saving data to Pyg file")
        torch.save(self.collate([d]), self.processed_paths[1])
        self.data, self.slices = self.collate([d])

    def describe(self):
        print("Pokec Pyg Dataset")
        print("Nodes:", len(self.data.x), "Edges:", len(self.data.edge_index[0]))