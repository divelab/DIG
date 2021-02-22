### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow


import numpy as np
import json


def one_hot(data, out_size=9, num_max_id=5):
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id))
    # data = data[data > 0]
    # 6 is C: Carbon, we adopt 6:C, 7:N, 8:O, 9:F only. the last place (4) is for padding virtual node.
    indices = np.where(data >= 6, data - 6, num_max_id - 1)
    b[np.arange(out_size), indices] = 1
    # print('[DEBUG] data', data, 'b', b)
    return b


def transform_fn(data):
    """

    :param data: ((9,), (4,9,9), (15,))
    :return:
    """
    node, adj, label = data   # node (9,), adj (4,9,9), label (15,)
    # convert to one-hot vector
    node = one_hot(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids(file_path):
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [int(idx)-1 for idx in data['valid_idxs']]
    return val_ids
