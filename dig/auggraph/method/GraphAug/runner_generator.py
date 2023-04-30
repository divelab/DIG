# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand (aanand@tamu.edu)

import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from .model import RewardGenModel
from dig.auggraph.dataset.aug_dataset import Subset, DegreeTrans
from .aug import Augmenter
from dig.auggraph.method.GraphAug.constants import *


class RunnerGenerator(object):
    r"""
    Runs the training of an augmented samples generator model which uses the
    already trained reward generation model. For a given graph, the model
    generates an augmented sample and a likelihood that this is a label
    invariant augmentation. This prediction is then evaluated by the reward
    generation model and a loss is computed based on these metrics. The loss
    is then minimized through training. Check
    :obj:`examples.auggraph.GraphAug.run_generator` for examples on how to
    run the generator model.

    Args:
        data_root_path (string): Directory where datasets should be saved.
        dataset_name (:class:`dig.auggraph.method.GraphAug.constants.enums.DatasetName`):
            Name of the graph dataset.
        conf (dict): Hyperparameters for the model. Check
            :obj:`examples.auggraph.GraphAug.conf.generator_conf` for
            examples on how to define the conf dictionary for the generator.
    """
    def __init__(self, data_root_path, dataset_name, conf):
        self.conf = conf
        self._get_dataset(data_root_path, dataset_name)
        self._get_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        if conf[BASELINE] == BaselineType.EXP:
            self.moving_mean = None

    def _get_dataset(self, data_root_path, dataset_name):
        dataset = TUDataset(data_root_path, name=dataset_name.value)
        if dataset_name in [DatasetName.NCI1, DatasetName.MUTAG, DatasetName.AIDS, DatasetName.NCI109, DatasetName.PROTEINS]:
            self.train_set = Subset(dataset)
            self.val_set = Subset(dataset)
        elif dataset_name in [DatasetName.COLLAB, DatasetName.IMDB_BINARY]:
            self.train_set = Subset(dataset, transform=DegreeTrans(dataset))
            self.val_set = Subset(dataset, transform=DegreeTrans(dataset))
            self.post_trans = DegreeTrans(dataset)
        in_dim = self.val_set[0].x.shape[1]
        self.conf[REWARD_GEN_PARAMS][IN_DIMENSION] = in_dim
        self.conf[GENERATOR_PARAMS][IN_DIMENSION] = in_dim
        if AugType.NODE_FM.value in self.conf[GENERATOR_PARAMS][AUG_TYPE_PARAMS]:
            self.conf[GENERATOR_PARAMS][AUG_TYPE_PARAMS][AugType.NODE_FM.value][NODE_FEAT_DIM] = in_dim

    def _get_model(self):
        self.reward_generator = RewardGenModel(**self.conf[REWARD_GEN_PARAMS])
        if self.conf[REWARD_GEN_STATE_PATH] is not None:
            self.reward_generator.load_state_dict(torch.load(self.conf[REWARD_GEN_STATE_PATH]))
        self.reward_generator.eval()
        self.generator = Augmenter(**self.conf[GENERATOR_PARAMS])

    def _train_epoch(self, loader, optimizer):
        self.generator.train()

        for iter, data_batch in enumerate(loader):
            anchor_data = data_batch
            anchor_data = anchor_data.to(self.device)
            g_loss_sum, reward_avg = 0, 0

            for _ in range(self.conf[GENERATOR_STEPS]):
                optimizer.zero_grad()

                neg_data, log_likelihoods = self.generator(anchor_data)
                if hasattr(self, 'post_trans'):
                    neg_data = self.post_trans(neg_data)

                neg_out = self.reward_generator(anchor_data, neg_data).view(-1)
                rewards = torch.log(neg_out)
                if self.conf[BASELINE] == BaselineType.MEAN:
                    baseline = torch.mean(rewards)
                    advantages = (rewards - baseline).detach()
                elif self.conf[BASELINE] == BaselineType.EXP:
                    new_mean = torch.mean(rewards)
                    if self.moving_mean is None:
                        self.moving_mean = new_mean.cpu().detach().item()
                    else:
                        new_mean = new_mean.cpu().detach().item()
                        self.moving_mean = self.conf[MOVING_RATIO] * self.moving_mean + (
                                    1.0 - self.conf[MOVING_RATIO]) * new_mean
                    advantages = (rewards - self.moving_mean).detach()
                elif isinstance(self.conf[BASELINE], float):
                    advantages = (rewards - self.conf[BASELINE]).detach()
                else:
                    advantages = rewards.detach()
                g_loss = -torch.sum(log_likelihoods.view(-1) * advantages)

                if torch.isnan(g_loss) or torch.isinf(g_loss):
                    torch.save(self.generator.state_dict(), os.path.join(self.out_path, 'buggy_model.pt'))
                    exit()

                g_loss.backward()
                optimizer.step()
                g_loss_sum += g_loss.detach().item()
                reward_avg += torch.mean(rewards).detach().item()

            print('Generator loss {}, Average reward {}'.format(g_loss_sum, reward_avg / self.conf[GENERATOR_STEPS]))

    def test(self):
        self.generator.eval()
        num_correct, total_rewards = 0, 0.0

        for data in self.val_set:
            data = data.to(self.device)

            aug_data, _ = self.generator(data)

            data1, data2 = Batch.from_data_list([data]), Batch.from_data_list([aug_data])
            output = self.reward_generator(data1, data2)
            pred = (output.view(-1) > 0.5).long()
            num_correct += pred.sum().item()
            total_rewards += torch.log(output).view(-1)[0].item()

        return num_correct / len(self.val_set), total_rewards / len(self.val_set)

    def train_test(self, results_path):
        r"""
        This method is used to run the training for the augmented samples
        generator and validate the epoch parameters.

        Args:
             results_path (string): Directory where the resulting optimal
                parameters of the generator model will be saved.
        """
        self.reward_generator, self.generator = self.reward_generator.to(self.device), self.generator.to(self.device)

        out_path = os.path.join(results_path, self.dataset_name.value)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        model_path = os.path.join(out_path, 'max_augs_{}'.format(self.conf[GENERATOR_PARAMS][MAX_NUM_AUG]))
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        self.out_path = out_path
        val_record_file = 'val_record.txt'
        f = open(os.path.join(model_path, val_record_file), 'a')
        f.write('Generator results for dataset {} using augmentation below\n'.format(self.dataset_name))
        for aug_type in self.conf[GENERATOR_PARAMS][AUG_TYPE_PARAMS]:
            f.write('{}: {}\n'.format(aug_type, self.conf[GENERATOR_PARAMS][AUG_TYPE_PARAMS][aug_type]))
        f.close()

        train_loader = DataLoader(self.train_set, batch_size=self.conf[BATCH_SIZE], shuffle=True, num_workers=8)

        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.conf[INITIAL_LR], weight_decay=1e-4)

        for epoch in range(self.conf[MAX_NUM_EPOCHS]):
            self._train_epoch(train_loader, optimizer)

            if epoch % self.conf[TEST_INTERVAL] == 0:
                val_acc, avg_reward = self.test()
                print('Epoch {}, validation accuracy {}, average reward {}'.format(epoch, val_acc, avg_reward))

                f = open(os.path.join(out_path, val_record_file), 'a')
                f.write('Epoch {}, validation accuracy {}, average reward {}\n'.format(epoch, val_acc, avg_reward))
                f.close()

            if self.conf[SAVE_MODEL]:
                torch.save(self.generator.state_dict(), os.path.join(model_path, '{}.pt'.format(str(epoch).zfill(4))))
