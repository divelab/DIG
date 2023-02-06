# Author: Youzhi Luo (yzluo@tamu.edu)
# Updated by: Anmol Anand(aanand@tamu.edu)

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from .model import DiscriminatorModel
from .utils import DegreeTrans, TripleSet
from .conf import dis_conf
from .constants import *


class RunnerDiscriminator(object):
    def __init__(self, dataset_name, data_root_path='tudataset/'):
        self.conf = dis_conf[dataset_name]
        self._get_dataset(data_root_path, dataset_name)
        self.model = self._get_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_name = dataset_name


    def _get_dataset(self, data_root_path, data_name):
        dataset = TUDataset(data_root_path, name=data_name)
        if data_name in [NCI1, MUTAG, PROTEINS, NCI109]:
            self.train_set = TripleSet(dataset)
            self.val_set = TripleSet(dataset)
        elif data_name in [COLLAB, IMDB_BINARY]:
            self.train_set = TripleSet(dataset, transform=DegreeTrans(dataset))
            self.val_set = TripleSet(dataset, transform=DegreeTrans(dataset))
        self.conf[DISCRIMINATOR_PARAMS][IN_DIMENSION] = self.train_set[0][0].x.shape[1]
    

    def _get_model(self):
        return DiscriminatorModel(**self.conf[DISCRIMINATOR_PARAMS])


    def _train_epoch(self, loader, optimizer):
        self.model.train()
        for data_batch in loader:
            anchor_data, pos_data, neg_data = data_batch
            anchor_data, pos_data, neg_data = anchor_data.to(self.device), pos_data.to(self.device), neg_data.to(self.device)

            optimizer.zero_grad()

            pos_out = self.model(anchor_data, pos_data).view(-1)
            pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))

            neg_out = self.model(anchor_data, neg_data).view(-1)
            neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()


    def test(self, loader):
        self.model.eval()
        num_correct, num_pos_correct, num_neg_correct = 0, 0, 0
        
        with torch.no_grad():
            for data_batch in loader:
                anchor_data, pos_data, neg_data = data_batch
                anchor_data, pos_data, neg_data = anchor_data.to(self.device), pos_data.to(self.device), neg_data.to(self.device)

                output = self.model(anchor_data, pos_data)
                pred = (output.view(-1) > 0.5).long()
                num_correct += pred.sum().item()
                num_pos_correct += pred.sum().item()
                
                output = self.model(anchor_data, neg_data)
                pred = (output.view(-1) < 0.5).long()
                num_correct += pred.sum().item()
                num_neg_correct += pred.sum().item()

        return num_correct / (2 * len(loader.dataset)), num_pos_correct / len(loader.dataset), num_neg_correct / len(loader.dataset)


    def train_test(self, out_root_path='results/dis_results', num_save=30, file_name='record.txt'):
        self.model = self.model.to(self.device)

        out_path = os.path.join(out_root_path, self.data_name)
        if not os.path.isdir(out_path):
            print(out_path)
            os.makedirs(out_path)

        model_dir_name = self.conf[DISCRIMINATOR_PARAMS][MODEL_TYPE]
        model_dir = os.path.join(out_path, model_dir_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        f = open(os.path.join(out_path, file_name), 'a')
        f.write('Discrimator classification results for dataset {} with model parameters {}\n'.format(self.data_name, 
            self.conf[DISCRIMINATOR_PARAMS]))
        f.close()

        train_loader = DataLoader(self.train_set, batch_size=self.conf[BATCH_SIZE], shuffle=True, num_workers=16)
        val_loader = DataLoader(self.val_set, batch_size=self.conf[BATCH_SIZE], shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf[INITIAL_LR], weight_decay=1e-4)

        for epoch in range(self.conf[MAX_NUM_EPOCHS]):
            self._train_epoch(train_loader, optimizer)
            if model_dir_name is not None and epoch > self.conf[MAX_NUM_EPOCHS] - num_save:
                torch.save(self.model.state_dict(), os.path.join(model_dir, '{}.pt'.format(str(epoch).zfill(4))))
            val_acc, val_pos_acc, val_neg_acc = self.test(val_loader)
            print('Epoch {}, validation accuracy {}, accuracy of positive samples {}, accuracy of negative samples {}'.format(epoch, val_acc, val_pos_acc, val_neg_acc))

            f = open(os.path.join(out_path, file_name), 'a')
            f.write('Epoch {}, validation accuracy {}, accuracy of positive samples {}, accuracy of negative samples {}\n'.format(epoch, val_acc, val_pos_acc, val_neg_acc))
            f.close()