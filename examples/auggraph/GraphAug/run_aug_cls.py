from dig.auggraph.method.GraphAug.runner_aug_cls import RunnerAugCls
from dig.auggraph.method.GraphAug.constants import *
from examples.auggraph.GraphAug.conf.generator_conf import generator_conf
from examples.auggraph.GraphAug.conf.aug_cls_conf import aug_cls_conf

dataset_name = DatasetName.IMDB_BINARY
conf = aug_cls_conf[dataset_name]
conf[GENERATOR_PARAMS] = generator_conf[dataset_name][GENERATOR_PARAMS]
generator_checkpoint = generator_conf[dataset_name][MAX_NUM_EPOCHS] - 1
conf[AUG_MODEL_PATH] = './dig/auggraph/method/GraphAug/results/generator_results/{}/max_augs_{}/{}.pt'.format(dataset_name.value, conf[GENERATOR_PARAMS][MAX_NUM_AUG], str(generator_checkpoint).zfill(4))

runner = RunnerAugCls('./dig/auggraph/datasets/tudatasets', dataset_name, conf)
for _ in range(5):
    runner.train_test('./dig/auggraph/method/GraphAug/results/aug_cls_results/{}'.format(conf[MODEL_NAME].value))
