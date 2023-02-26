from dig.auggraph.method.GraphAug.constants import *

DATA_ROOT_PATH = 'dig/auggraph/datasets/tudatasets'

REWARD_GEN_RESULTS_PATH = 'dig/auggraph/method/GraphAug/results/reward_gen_results'

GENERATOR_RESULTS_PATH = 'dig/auggraph/method/GraphAug/results/generator_results'

def get_reward_gen_state_path(dataset_name, model_type, checkpoint):
    return REWARD_GEN_RESULTS_PATH + '/{}/{}/{}.pt'.format(dataset_name.value, model_type.value, str(checkpoint).zfill(4))

