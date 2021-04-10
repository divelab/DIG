### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow


from logging import getLogger

import numpy
from rdkit import Chem
from tqdm import tqdm
import sys
# for linux env.
sys.path.insert(0,'..')
from preprocess_data.smile_to_graph import GGNNPreprocessor,MolFeatureExtractionError
from preprocess_data.data_loader import NumpyTupleDataset

import traceback
# Code adapted from chainer_chemistry\dataset\parsers\data_frame_parser.py


class DataFrameParser(object):
    """data frame parser

    This FileParser parses pandas dataframe.
    It should contain column which contain SMILES as input, and
    label column which is the target to predict.

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
        labels (str or list or None): labels column
        smiles_col (str): smiles column
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
        logger:
    """

    def __init__(self, preprocessor,
                 labels=None,
                 smiles_col='smiles',
                 postprocess_label=None, postprocess_fn=None,
                 logger=None):
        super(DataFrameParser, self).__init__()
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels  # type: list
        self.smiles_col = smiles_col
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)
        self.preprocessor = preprocessor

    def parse(self, df, return_smiles=False, target_index=None,
              return_is_successful=False):
        """parse DataFrame using `preprocessor`

        Label is extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.

        Args:
            df (pandas.DataFrame): dataframe to be parsed.
            return_smiles (bool): If set to `True`, smiles list is returned in
                the key 'smiles', it is a list of SMILES from which input
                features are successfully made.
                If set to `False`, `None` is returned in the key 'smiles'.
            target_index (list or None): target index list to partially extract
                dataset. If None (default), all examples are parsed.
            return_is_successful (bool): If set to `True`, boolean list is
                returned in the key 'is_successful'. It represents
                preprocessing has succeeded or not for each SMILES.
                If set to False, `None` is returned in the key 'is_success'.

        Returns (dict): dictionary that contains Dataset, 1-d numpy array with
            dtype=object(string) which is a vector of smiles for each example
            or None.

        """
        logger = self.logger
        pp = self.preprocessor
        smiles_list = []
        is_successful_list = []

        # counter = 0
        if isinstance(pp, GGNNPreprocessor):
            if target_index is not None:
                df = df.iloc[target_index]

            features = None
            smiles_index = df.columns.get_loc(self.smiles_col)
            if self.labels is None:
                labels_index = []  # dummy list
            else:
                labels_index = [df.columns.get_loc(c) for c in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0
            for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
                smiles = row[smiles_index]
                # TODO(Nakago): Check.
                # currently it assumes list
                labels = [row[i] for i in labels_index]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        fail_count += 1
                        if return_is_successful:
                            is_successful_list.append(False)
                        continue
                    # Note that smiles expression is not unique.
                    # we obtain canonical smiles
                    canonical_smiles, mol = pp.prepare_smiles_and_mol(mol)
                    input_features = pp.get_input_features(mol)

                    # Extract label
                    if self.postprocess_label is not None:
                        labels = self.postprocess_label(labels)

                    if return_smiles:
                        smiles_list.append(canonical_smiles)
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                except Exception as e:
                    logger.warning('parse(), type: {}, {}'
                                   .format(type(e).__name__, e.args))
                    logger.info(traceback.format_exc())
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                # Initialize features: list of list
                if features is None:
                    if isinstance(input_features, tuple):
                        num_features = len(input_features)
                    else:
                        num_features = 1
                    if self.labels is not None:
                        num_features += 1
                    features = [[] for _ in range(num_features)]

                if isinstance(input_features, tuple):
                    for i in range(len(input_features)):
                        features[i].append(input_features[i])
                else:
                    features[0].append(input_features)
                if self.labels is not None:
                    features[len(features) - 1].append(labels)
                success_count += 1
                if return_is_successful:
                    is_successful_list.append(True)
            ret = []

            for feature in features:
                try:
                    feat_array = numpy.asarray(feature)
                except ValueError:
                    # Temporal work around.
                    # See,
                    # https://stackoverflow.com/questions/26885508/why-do-i-get-error-trying-to-cast-np-arraysome-list-valueerror-could-not-broa
                    feat_array = numpy.empty(len(feature), dtype=numpy.ndarray)
                    feat_array[:] = feature[:]
                ret.append(feat_array)
            result = tuple(ret)
            logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                        .format(fail_count, success_count, total_count))
        else:
            raise NotImplementedError

        smileses = numpy.array(smiles_list) if return_smiles else None
        if return_is_successful:
            is_successful = numpy.array(is_successful_list)
        else:
            is_successful = None

        # if isinstance(result, tuple):
        #     if self.postprocess_fn is not None:
        #         result = self.postprocess_fn(*result)
        #     dataset = NumpyTupleDataset(*result)
        # else:
        #     if self.postprocess_fn is not None:
        #         result = self.postprocess_fn(result)
        #     dataset = NumpyTupleDataset(result)

        if isinstance(result, (tuple, list)):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            dataset = NumpyTupleDataset(result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            dataset = NumpyTupleDataset([result])

        return {"dataset": dataset,
                "smiles": smileses,
                "is_successful": is_successful}

    def extract_total_num(self, df):
        """Extracts total number of data which can be parsed

        We can use this method to determine the value fed to `target_index`
        option of `parse` method. For example, if we want to extract input
        feature from 10% of whole dataset, we need to know how many samples
        are in a file. The returned value of this method may not to be same as
        the final dataset size.

        Args:
            df (pandas.DataFrame): dataframe to be parsed.

        Returns (int): total number of dataset can be parsed.

        """
        return len(df)
