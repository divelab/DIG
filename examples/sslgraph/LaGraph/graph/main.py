import os
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time

from dataset import TUDatasetLaGraph
from torch_geometric.data import DataLoader

from model import LaGraphNet
from loss import LaGraphLoss
from evaluate_embedding import evaluate_embedding

from arguments import arg_parse_graph
import random
from tqdm import tqdm


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    

if __name__ == '__main__':
    args = arg_parse_graph()
    path = osp.join(args.data_dir, args.DS)
    accuracies = []
    val_accuracies = []

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    txtfile = args.save_dir+'record.txt'

    if txtfile is not None:
        file_obj = open(txtfile, 'w')
        file_obj.write('================\n')
        file_obj.write('args: {}\n'.format(args))
        file_obj.write('================\n')
        file_obj.close()

    for seed in range(5):
        setup_seed(seed)
        
        # Dataset & dataloader
        pretrain_dataset = TUDatasetLaGraph(path, name=args.DS, aug=True, args=args).shuffle()
        dataset = TUDatasetLaGraph(path, name=args.DS, aug=False, args=args).shuffle()
        try:
            dataset_num_features = dataset.get_num_feature()
        except:
            dataset_num_features = 1
            
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.bsz)
        loader = DataLoader(dataset, batch_size=args.bsz)
        
        # Define model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LaGraphNet(dim=dataset_num_features, num_en_layers=args.enly, num_de_layers=args.dely, pool=args.pool, decoder=args.decoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if txtfile is not None:
            file_obj = open(txtfile, 'a')
            file_obj.write('================\n')
            file_obj.write('seed: {}\n'.format(seed))
            file_obj.write('================\n')
            file_obj.close()

        accuracy = []
        val_accuracy = []

        for epoch in tqdm(range(1, args.epoch+1), total=args.epoch, desc="Seed: {}; Epoch".format(seed), position=0, leave=True):
            loss_all, loss_rec_all, loss_inv_all = 0, 0, 0
            num_nodes_all = 0
            model.train()
            for data in pretrain_loader:
                optimizer.zero_grad()
                data, data_aug = data
                data, data_aug = data.to(device), data_aug.to(device)

                _, dec_orig, enc_orig_g = model(data.x, data.edge_index, data.batch)
                _, _, enc_aug_g = model(data_aug.x, data_aug.edge_index, data_aug.batch)
                loss, loss_rec, loss_inv = LaGraphLoss()(data.x, enc_orig_g, dec_orig, enc_aug_g, data_aug.mask, args.loss, args.alpha, dataset_num_features)

                loss_all += loss.item() * data.num_nodes
                loss_rec_all += loss_rec.item() * data.num_nodes
                loss_inv_all += loss_inv.item() * data.num_nodes
                num_nodes_all += data.num_nodes             

                loss.backward()
                optimizer.step()

            if txtfile is not None:
                file_obj = open(txtfile, 'a')
                file_obj.write('Epoch {}, Loss {}, Loss_rec {}, Loss_inv {}\n'
                               .format(epoch, loss_all / num_nodes_all, loss_rec_all / num_nodes_all, loss_inv_all / num_nodes_all))
                file_obj.close()

            if epoch % args.interval == 0:
                if args.save_model:
                    model.save_network(args.save_dir+'net_seed'+str(seed)+'ep'+str(epoch)+'.pth')

                model.eval()
                global_reps, y = model.get_global_rep(loader)
                val_acc, acc = evaluate_embedding(global_reps, y, mode='test')
                accuracy.append(acc*100)
                
                if txtfile is not None:
                    file_obj = open(txtfile, 'a')
                    file_obj.write('Epoch {}; test_acc {}\n'.format(epoch, acc))
                    file_obj.close()

        accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)

    accuracies = np.array(accuracies)
    avg_accuracies = np.mean(accuracies, axis=0)
    std_accuracies = np.std(accuracies, axis=0)

    if txtfile is not None:
        file_obj = open(txtfile, 'a')
        file_obj.write('==========\n')
        for i in range(int(args.epoch/args.interval)):
            file_obj.write('Epoch {0:d}, test_accs {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.2f}, test_acc_avg {6:.2f}, test_acc_std {7:.2f}\n'
                               .format((i+1)*args.interval, accuracies[0][i], accuracies[1][i], accuracies[2][i], accuracies[3][i], accuracies[4][i],
                                       avg_accuracies[i], std_accuracies[i]))
#             print('Epoch {0:d}, test_accs {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.2f}, test_acc_avg {6:.2f}, test_acc_std {7:.2f}'
#                         .format((i+1)*args.interval, accuracies[0][i], accuracies[1][i], accuracies[2][i], accuracies[3][i], accuracies[4][i],
#                                 avg_accuracies[i], std_accuracies[i]))

    file_obj.write('==========\n')
    best_epoch = np.argmax(avg_accuracies) + 1
    file_obj.write('Best epoch: {0:d}, avg_acc: {1:.2f}, acc_std: {2:.2f}\n'.format(best_epoch, avg_accuracies[best_epoch-1], std_accuracies[best_epoch-1]))
    file_obj.close()
    print('==========')
    print('Best epoch: {0:d}, avg_acc: {1:.2f}, acc_std: {2:.2f}\n'.format(best_epoch, avg_accuracies[best_epoch-1], std_accuracies[best_epoch-1]))
    



