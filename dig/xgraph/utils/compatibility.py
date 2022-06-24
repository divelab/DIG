import re
from collections import OrderedDict

from torch_geometric import __version__


def compatible_state_dict(state_dict):
    comp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        comp_key = key
        comp_value = value
        if int(__version__[0]) >= 2:
            comp_key = re.sub(r'conv(1|s.[0-9]).weight', 'conv\g<1>.lin.weight', key)
            if comp_key != key:
                comp_value = value.T
        if comp_key != key:
            comp_state_dict[key] = value
        comp_state_dict[comp_key] = comp_value
    return comp_state_dict
