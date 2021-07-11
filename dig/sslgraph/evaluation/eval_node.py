import copy
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import DataLoader
from sklearn import preprocessing


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


class NodeUnsupervised(object):
    r"""
    The evaluation interface for unsupervised graph representation learning evaluated with 
    linear classification. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph>`_ 
    for examples of usage.
    
    Args:
        full_dataset (torch_geometric.data.Dataset): The graph classification dataset.
        train_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for training. Set to :obj:`None` if included in dataset.
            (default: :obj:`None`)
        val_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for validation. Set to :obj:`None` if included in dataset.
            (default: :obj:`None`)
        test_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for test. Set to :obj:`None` if included in dataset. (default: :obj:`None`)
        classifier (string, optional): Linear classifier for evaluation, :obj:`"SVC"` or 
            :obj:`"LogReg"`. (default: :obj:`"LogReg"`)
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> node_dataset = get_node_dataset("Cora") # using default train/test split
    >>> evaluator = NodeUnsupervised(node_dataset, log_interval=10, device=0)
    >>> evaluator.evaluate(model, encoder)
    
    >>> node_dataset = SomeDataset()
    >>> # Using your own dataset or with different train/test split
    >>> train_mask, val_mask, test_mask = torch.Tensor([...]), torch.Tensor([...]), torch.Tensor([...])
    >>> evaluator = NodeUnsupervised(node_dataset, train_mask, val_mask, test_mask, log_interval=10, device=0)
    >>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, full_dataset, train_mask=None, val_mask=None, test_mask=None, 
                 classifier='LogReg', metric='acc', device=None, log_interval=1, **kwargs):

        self.full_dataset = full_dataset
        self.train_mask = full_dataset[0].train_mask if train_mask is None else train_mask
        self.val_mask = full_dataset[0].val_mask if val_mask is None else val_mask
        self.test_mask = full_dataset[0].test_mask if test_mask is None else test_mask
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
        self.setup_train_config(**kwargs)

    def setup_train_config(self, p_optim = 'Adam', p_lr = 0.01, p_weight_decay = 0, 
                           p_epoch = 2000, logreg_wd = 0, comp_embed_on='cpu'):

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.comp_embed_on = comp_embed_on
        self.logreg_wd = logreg_wd

    def evaluate(self, learning_model, encoder):
        r"""Run evaluation with given learning model and encoder(s).
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive)
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.

        :rtype: (float, float)
        """
        
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
    
    
    def evaluate_multisplits(self, learning_model, encoder, split_masks):
        r"""Run evaluation with given learning model and encoder(s), return averaged scores 
        on multiple different splits.
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive)
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            split_masks (list, or generator): A list of generator that contains or yields masks for
                train, val and test splits.

        :rtype: float

        Example
        -------
        >>> split_masks = [(train1, val1, test1), (train2, val2, test2), ..., (train20, val20, test20)]
        """
        
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
                for train_mask, val_mask, test_mask in split_masks:
                    test_score = self.get_clf()(embed[train_mask], lbls[train_mask],
                                                embed[test_mask], lbls[test_mask])
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


    def grid_search(self, learning_model, encoder, p_lr_lst=[0.1,0.01,0.001], 
                    p_epoch_lst=[2000]):
        r"""Perform grid search on learning rate and epochs in pretraining.
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive) 
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            p_lr_lst (list, optional): List of learning rate candidates.
            p_epoch_lst (list, optional): List of epochs number candidates.

        :rtype: (float, float, (float, int))
        """
        
        acc_m_lst = []
        acc_sd_lst = []
        paras = []
        for p_lr in p_lr_lst:
            for p_epoch in p_epoch_lst:
                self.setup_train_config(p_lr=p_lr, p_epoch=p_epoch)
                model = copy.deepcopy(learning_model)
                enc = copy.deepcopy(encoder)
                acc_m, acc_sd = self.evaluate(model, enc)
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

