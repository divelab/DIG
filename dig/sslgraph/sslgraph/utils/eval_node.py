import sys, copy, torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
from torch import optim
from sklearn import preprocessing
from sslgraph.utils import get_node_dataset


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


class EvalUnsupevised(object):
    def __init__(self, full_dataset, train_mask, val_mask, test_mask, 
                 classifier='LogReg', metric='acc', device=None, log_interval=1):

        self.full_dataset = full_dataset
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.metric = metric
        self.device = device
        self.classifier = classifier
        self.log_interval = log_interval
        self.num_classes = full_dataset.num_classes
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        # Use default config if not further specified
        self.setup_train_config()

    def setup_train_config(self, p_optim = 'Adam', p_lr = 0.01, p_weight_decay = 0, 
                           p_epoch = 2000, logreg_wd = 0, comp_embed_on='cpu'):

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.comp_embed_on = comp_embed_on
        self.logreg_wd = logreg_wd

    def evaluate(self, learning_model, encoder, pred_head=None, fold_seed=None):
        '''
        Args:
            learning_model: An object of a contrastive model or a predictive model.
            encoder: List or trainable pytorch model.
            pred_head: [Optional] Trainable pytoch model. If None, will use linear projection.
        '''
        
        full_loader = DataLoader(self.full_dataset, 1)
        if isinstance(encoder, list):
            params = [{'params': enc.parameters()} for enc in encoder]
        else:
            params = encoder.parameters()
        
        p_optimizer = self.get_optim(self.p_optim)(params, lr=self.p_lr, 
                                                   weight_decay=self.p_weight_decay)

        test_scores_m, test_scores_sd = [], []
        per_epoch_out = (self.log_interval<self.p_epoch)
        for i, enc in enumerate(learning_model.train(encoder, full_loader, 
                                                     p_optimizer, self.p_epoch, per_epoch_out)):
            if not per_epoch_out or (i+1)%self.log_interval==0:
                embed, lbls = self.get_embed(enc.to(self.device), full_loader)
                lbs = np.array(preprocessing.LabelEncoder().fit_transform(lbls))
                
                test_scores = []
                for _ in range(10):
                    test_score = self.get_clf()(embed[self.train_mask], lbls[self.train_mask],
                                                embed[self.test_mask], lbls[self.test_mask])
                    test_scores.append(test_score)
                
                test_scores = torch.tensor(test_scores)
                test_score_mean = test_scores.mean().item()
                test_score_std = test_scores.std().item() 
                test_scores_m.append(test_score_mean)
                test_scores_sd.append(test_score_std)
                
        idx = np.argmax(test_scores_m)
        acc = test_scores_m[idx]
        std = test_scores_sd[idx]
        print('Best epoch %d: acc %.4f (+/- %.4f).'%((idx+1)*self.log_interval, acc, std))
        return acc


    def grid_search(self, learning_model, encoder, pred_head=None, fold_seed=12345,
                    p_lr_lst=[0.1,0.01,0.001], p_epoch_lst=[2000]):
        
        acc_m_lst = []
        acc_sd_lst = []
        paras = []
        for p_lr in p_lr_lst:
            for p_epoch in p_epoch_lst:
                self.setup_train_config(p_lr=p_lr, p_epoch=p_epoch)
                model = copy.deepcopy(learning_model)
                enc = copy.deepcopy(encoder)
                acc_m, acc_sd = self.evaluate(model, enc, pred_head, fold_seed)
                acc_m_lst.append(acc_m)
                acc_sd_lst.append(acc_sd)
                paras.append((p_lr, p_epoch))
        idx = np.argmax(acc_m_lst)
        print('Best paras: %d epoch, lr=%f, acc=%.4f' %(
            paras[idx][1], paras[idx][0], acc_m_lst[idx]))
        
        return acc_m_lst[idx], acc_sd_lst[idx], paras[idx]

    
    def svc_clf(self, train_embs, train_lbls, test_embs, test_lbls):

        if self.search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)

        classifier.fit(train_embs, train_lbls)
        acc = accuracy_score(test_lbls, classifier.predict(test_embs))
            
        return acc
    
    
    def log_reg(self, train_embs, train_lbls, test_embs, test_lbls):
        
        hid_units = train_embs.shape[1]
        train_embs = torch.from_numpy(train_embs).to(self.device)
        train_lbls = torch.from_numpy(train_lbls).to(self.device)
        test_embs = torch.from_numpy(test_embs).to(self.device)
        test_lbls = torch.from_numpy(test_lbls).to(self.device)

        xent = nn.CrossEntropyLoss()
        log = LogReg(hid_units, self.num_classes)
        log.to(self.device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, 
                               weight_decay=self.logreg_wd)

        best_val = 0
        test_acc = None
        for it in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        
        return acc.item()
    
    
    def get_embed(self, model, loader):
    
        model.eval()
        model.to(self.comp_embed_on)
        ret, y = [], []
        with torch.no_grad():
            for data in loader:
                y.append(data.y.numpy())
                data.to(self.comp_embed_on)
                embed = model(data)
                ret.append(embed.cpu().numpy())
                
        model.to(self.device)
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
        
        
    def get_clf(self):
        
        if self.classifier == 'SVC':
            return self.svc_clf
        elif self.classifier == 'LogReg':
            return self.log_reg
        else:
            return None
        
    
    def get_optim(self, optim):
        
        optims = {'Adam': torch.optim.Adam}
        
        return optims[optim]
    
