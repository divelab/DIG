import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class graphair(nn.Module):
    r'''
        Implementation of Graphair from the paper `"LEARNING FAIR GRAPH REPRESENTATIONS VIA AUTOMATED DATA AUGMENTATIONS`"
    '''
    def __init__(self, aug_model, f_encoder, sens_model, lr = 1e-4, weight_decay = 1e-5, alpha = 20, beta = 0.9, gamma = 0.7, lam = 1, dataset = 'POKEC', batch_size = None, num_hidden = 64, num_proj_hidden = 64):
        super(graphair, self).__init__()
        self.aug_model = aug_model
        self.f_encoder = f_encoder
        self.sens_model = sens_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dataset = dataset
        self.lam = lam

        self.criterion_sens = nn.BCEWithLogitsLoss()
        self.criterion_cont= nn.CrossEntropyLoss()
        self.criterion_recons = nn.MSELoss()

        self.optimizer_s = torch.optim.Adam(self.sens_model.parameters(), lr = 1e-4, weight_decay = 1e-5)

        FG_params = [{'params': self.aug_model.parameters(), 'lr': 1e-4} ,  {'params':self.f_encoder.parameters()}]
        self.optimizer = torch.optim.Adam(FG_params, lr = lr, weight_decay = weight_decay)

        self.optimizer_aug = torch.optim.Adam(self.aug_model.parameters(), lr = 1e-3, weight_decay = weight_decay)
        self.optimizer_enc = torch.optim.Adam(self.f_encoder.parameters(), lr = lr, weight_decay = weight_decay)

        self.batch_size = batch_size

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
    
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def info_nce_loss_2views(self, features):
        
        batch_size = int(features.shape[0] / 2)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        temperature = 0.07
        logits = logits / temperature
        return logits, labels


    def forward(self, adj, x):
        pass
    
    def fit_batch_GraphSAINT(self,epoches, adj, x , sens, idx_sens, minibatch, warmup = None, adv_epoches = 10):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = sp.csr_matrix(adj)
        norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)
        
        idx_sens = idx_sens.cpu().numpy()

        if warmup:
            for _ in range(warmup):

                node_subgraph, adj, _ = minibatch.one_batch(mode='train')
                adj = adj.cuda()
                edge_label = torch.FloatTensor(adj_orig[node_subgraph][:,node_subgraph].toarray()).cuda()

                adj_aug, x_aug, adj_logits = self.aug_model(adj, x[node_subgraph], adj_orig = edge_label)
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, edge_label)


                feat_loss =  self.criterion_recons(x_aug, x[node_subgraph])
                recons_loss =  edge_loss + self.beta * feat_loss

                self.optimizer_aug.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    recons_loss.backward(retain_graph=True)
                self.optimizer_aug.step()

                print(
                'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                )

        for epoch_counter in range(epoches):
            ### generate fair view
            node_subgraph, adj, norm_loss_subgraph = minibatch.one_batch(mode='train')
            adj = adj.cuda()
            norm_loss_subgraph = norm_loss_subgraph.cuda()

            edge_label = torch.FloatTensor(adj_orig[node_subgraph][:,node_subgraph].toarray()).cuda()
            adj_aug, x_aug, adj_logits = self.aug_model(adj, x[node_subgraph], adj_orig = edge_label)
            # print("aug done")

            ### extract node representations
            h = self.projection(self.f_encoder(adj, x[node_subgraph]))
            h_prime = self.projection(self.f_encoder(adj_aug, x_aug))
            # print("encoder done")

            ### update sens model
            adj_aug_nograd = adj_aug.detach()
            x_aug_nograd = x_aug.detach()

            mask = np.in1d(node_subgraph, idx_sens)

            if (epoch_counter == 0):
                sens_epoches = adv_epoches * 10
            else:
                sens_epoches = adv_epoches
            for _ in range(sens_epoches):

                s_pred , _  = self.sens_model(adj_aug_nograd, x_aug_nograd)
                senloss = torch.nn.BCEWithLogitsLoss(weight=norm_loss_subgraph,reduction='sum')(s_pred[mask].squeeze(),sens[node_subgraph][mask].float())

                self.optimizer_s.zero_grad()
                senloss.backward()
                self.optimizer_s.step()

            s_pred , _  = self.sens_model(adj_aug, x_aug)
            senloss = torch.nn.BCEWithLogitsLoss(weight=norm_loss_subgraph,reduction='sum')(s_pred[mask].squeeze(),sens[node_subgraph][mask].float())

            ## update aug model
            logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim = 0))
            contrastive_loss = (torch.nn.CrossEntropyLoss(reduction='none')(logits, labels) * norm_loss_subgraph.repeat(2)).sum() 

            ## update encoder
            edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, edge_label)


            feat_loss =  self.criterion_recons(x_aug, x[node_subgraph])
            recons_loss =  edge_loss + self.lam * feat_loss

            loss = self.beta * contrastive_loss + self.gamma * recons_loss - self.alpha * senloss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if ((epoch_counter + 1) % 1000 == 0):
                print('Epoch: {:04d}'.format(epoch_counter+1),
                'sens loss: {:.4f}'.format(senloss.item()),
                'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
                'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                )

        torch.save(self.state_dict(),"./checkpoint/graphair_{}_alpha{}_beta{}_gamma{}_lambda{}_batch_size{}".format(self.dataset, self.alpha, self.beta, self.gamma, self.lam, self.batch_size))
