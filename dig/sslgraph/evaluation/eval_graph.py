import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

    
class GraphUnsupervised(object):
    r"""
    The evaluation interface for unsupervised graph representation learning evaluated with 
    linear classification. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph>`_ 
    for examples of usage.
    
    Args:
        dataset (torch_geometric.data.Dataset): The graph classification dataset.
        classifier (string, optional): Linear classifier for evaluation, :obj:`"SVC"` or 
            :obj:`"LogReg"`. (default: :obj:`"SVC"`)
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        epoch_select (string, optional): :obj:`"test_max"` or :obj:`"val_max"`.
            (default: :obj:`"test_max"`)
        n_folds (int, optional): Number of folds for evaluation. (default: :obj:`10`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> encoder = Encoder(...)
    >>> model = Contrastive(...)
    >>> evaluator = GraphUnsupervised(dataset, log_interval=10, device=0, p_lr = 0.001)
    >>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, dataset, classifier='SVC', log_interval=1, epoch_select='test_max', 
                 metric='acc', n_folds=10, device=None, **kwargs):
        
        self.dataset = dataset
        self.epoch_select = epoch_select
        self.metric = metric
        self.classifier = classifier
        self.log_interval = log_interval
        self.n_folds = n_folds
        self.out_dim = dataset.num_classes
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        # Use default config if not further specified
        self.setup_train_config(**kwargs)

    def setup_train_config(self, batch_size = 256, p_optim = 'Adam', p_lr = 0.01, 
                           p_weight_decay = 0, p_epoch = 20, svc_search = True):
        r"""Method to setup training config.
        
        Args:
            batch_size (int, optional): Batch size for pretraining and inference. 
                (default: :obj:`256`)
            p_optim (string, or torch.optim.Optimizer class): Optimizer for pretraining.
                (default: :obj:`"Adam"`)
            p_lr (float, optional): Pretraining learning rate. (default: :obj:`0.01`)
            p_weight_decay (float, optional): Pretraining weight decay rate. 
                (default: :obj:`0`)
            p_epoch (int, optional): Pretraining epochs number. (default: :obj:`20`)
            svc_search (string, optional): If :obj:`True`, search for hyper-parameter 
                :obj:`C` in SVC. (default: :obj:`True`)
        """
        
        self.batch_size = batch_size

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.search = svc_search
    
    def evaluate(self, learning_model, encoder, fold_seed=None):
        r"""Run evaluation with given learning model and encoder(s).
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive) 
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            fold_seed (int, optional): Seed for fold split. (default: :obj:`None`)

        :rtype: (float, float)
        """
        
        pretrain_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        if isinstance(encoder, list):
            params = [{'params': enc.parameters()} for enc in encoder]
        else:
            params = encoder.parameters()
        
        p_optimizer = self.get_optim(self.p_optim)(params, lr=self.p_lr, 
                                                   weight_decay=self.p_weight_decay)
        
        test_scores_m, test_scores_sd = [], []
        for i, enc in enumerate(learning_model.train(encoder, pretrain_loader, 
                                                     p_optimizer, self.p_epoch, True)):
            if (i+1)%self.log_interval==0:
                test_scores = []
                loader = DataLoader(self.dataset, self.batch_size, shuffle=False)
                embed, lbls = self.get_embed(enc.to(self.device), loader)
                lbs = np.array(preprocessing.LabelEncoder().fit_transform(lbls))

                kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=fold_seed)
                for fold, (train_index, test_index) in enumerate(kf.split(embed, lbls)):
                    test_score = self.get_clf()(embed[train_index], lbls[train_index],
                                                embed[test_index], lbls[test_index])
                    test_scores.append(test_score)

                kfold_scores = torch.tensor(test_scores)
                test_score_mean = kfold_scores.mean().item()
                test_score_std = kfold_scores.std().item() 
                test_scores_m.append(test_score_mean)
                test_scores_sd.append(test_score_std)
        
        idx = np.argmax(test_scores_m)
        acc = test_scores_m[idx]
        sd = test_scores_sd[idx]
        print('Best epoch %d: acc %.4f +/-(%.4f)'%((idx+1)*self.log_interval, acc, sd))
        return acc, sd 


    def grid_search(self, learning_model, encoder, fold_seed=12345,
                    p_lr_lst=[0.1,0.01,0.001], p_epoch_lst=[20,40,60]):
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
                acc_m, acc_sd = self.evaluate(model, enc, fold_seed)
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
        
        train_embs = torch.from_numpy(train_embs).to(self.device)
        train_lbls = torch.from_numpy(train_lbls).to(self.device)
        test_embs = torch.from_numpy(test_embs).to(self.device)
        test_lbls = torch.from_numpy(test_lbls).to(self.device)

        xent = nn.CrossEntropyLoss()
        log = LogReg(hid_units, nb_classes)
        log.to(self.device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        
        return acc
    
    
    def get_embed(self, model, loader):
    
        model.eval()
        ret, y = [], []
        with torch.no_grad():
            for data in loader:
                y.append(data.y.numpy())
                data.to(self.device)
                embed = model(data)
                ret.append(embed.cpu().numpy())

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
        
        if callable(optim):
            return optim
        
        optims = {'Adam': torch.optim.Adam}
        
        return optims[optim]
    

    
class PredictionModel(nn.Module):
    
    def __init__(self, encoder, pred_head, dim, out_dim):
        
        super(PredictionModel, self).__init__()
        self.encoder = encoder
        
        if pred_head is not None:
            self.pred_head = pred_head
        else:
            self.pred_head = nn.Linear(dim, out_dim)
        
    def forward(self, data):
        
        zg = self.encoder(data)
        out = self.pred_head(zg)
        
        return nn.functional.log_softmax(out, dim=-1)

    

class GraphSemisupervised(object):
    r"""
    The evaluation interface for semi-supervised learning and transfer learning for 
    graph-level tasks with pretraining and finetuning datasets. You can refer to `the benchmark 
    code <https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph>`_ for examples of usage.
    
    Args:
        dataset (torch_geometric.data.Dataset): The graph dataset for finetuning and evaluation.
        dataset_pretrain (torch_geometric.data.Dataset): The graph dataset for pretraining.
        label_rate (float, optional): Ratio of labels to use in finetuning dataset.
            (default: :obj:`1`)
        epoch_select (string, optional): :obj:`"test_max"` or :obj:`"val_max"`.
            (default: :obj:`"test_max"`)
        n_folds (int, optional): Number of folds for evaluation. (default: :obj:`10`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> dataset, pretrain_dataset = get_dataset("NCI1", "semisupervised")
    >>> evaluator = GraphSemisupervised(dataset, pretrain_dataset, device=0)
    >>> evaluator.evaluate(model, encoder) # semi-supervised learning
    
    >>> dataset = MoleculeNet("./transfer_data", "HIV")
    >>> pretrain_dataset = ZINC("./transfer_data")
    >>> evaluator = GraphSemisupervised(dataset, pretrain_dataset, device=0)
    >>> evaluator.evaluate(model, encoder) # transfer learning for molecule classification
    
    Note
    ----
    When using :obj:`torch_geometric.data.Dataset` without our provided :obj:`get_dataset`
    function, you may need to manually add self-loops before input to evaluator if some view 
    function requires them, such as diffusion.
    """
    
    def __init__(self, dataset, dataset_pretrain, label_rate=1, loss=nn.functional.nll_loss, 
                 epoch_select='test_max', metric='acc', n_folds=10, device=None, **kwargs):
        
        self.dataset, self.dataset_pretrain = dataset, dataset_pretrain
        self.label_rate = label_rate
        self.out_dim = dataset.num_classes
        self.metric = metric
        self.n_folds = n_folds
        self.loss = loss
        self.epoch_select = epoch_select
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Use default config if not further specified
        self.setup_train_config(**kwargs)
        
        
    def setup_train_config(self, batch_size = 128,
                           p_optim = 'Adam', p_lr = 0.0001, p_weight_decay = 0, p_epoch = 100,
                           f_optim = 'Adam', f_lr = 0.001, f_weight_decay = 0, f_epoch = 100):
        r"""Method to setup training config.
        
        Args:
            batch_size (int, optional): Batch size for pretraining and inference. 
                (default: :obj:`128`)
            p_optim (string, or torch.optim.Optimizer class): Optimizer for pretraining.
                (default: :obj:`"Adam"`)
            p_lr (float, optional): Pretraining learning rate. (default: :obj:`0.0001`)
            p_weight_decay (float, optional): Pretraining weight decay rate. 
                (default: :obj:`0`)
            p_epoch (int, optional): Pretraining epochs number. (default: :obj:`100`)
            f_optim (string, or torch.optim.Optimizer class): Optimizer for finetuning.
                (default: :obj:`"Adam"`)
            f_lr (float, optional): Finetuning learning rate. (default: :obj:`0.001`)
            f_weight_decay (float, optional): Finetuning weight decay rate. 
                (default: :obj:`0`)
            f_epoch (int, optional): Finetuning epochs number. (default: :obj:`100`)
        """
        
        self.batch_size = batch_size
        
        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.f_optim = f_optim
        self.f_lr = f_lr
        self.f_weight_decay = f_weight_decay
        self.f_epoch = f_epoch
        
    
    def evaluate(self, learning_model, encoder, pred_head=None, fold_seed=12345):
        r"""Run evaluation with given learning model and encoder(s).
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive)
                or a predictive model.
            encoder (torch.nn.Module, or list): Trainable pytorch model or list of models.
            pred_head (torch.nn.Module, optional): Prediction head. If None, will use linear 
                projection. (default: :obj:`None`)

        :rtype: (float, float)
        """
        pretrain_loader = DataLoader(self.dataset_pretrain, self.batch_size, shuffle=True)
        p_optimizer = self.get_optim(self.p_optim)(encoder.parameters(), lr=self.p_lr,
                                                   weight_decay=self.p_weight_decay)
        if self.p_epoch > 0:
            encoder = next(learning_model.train(encoder, pretrain_loader, p_optimizer, self.p_epoch))
        model = PredictionModel(encoder, pred_head, learning_model.z_dim, self.out_dim).to(self.device)
        
        test_scores = []
        val_losses = []
        val = not (self.epoch_select == 'test_max' or self.epoch_select == 'test_min')
        
        for fold, train_loader, test_loader, val_loader in k_fold(
            self.n_folds, self.dataset, self.batch_size, self.label_rate, val, fold_seed):
            
            fold_model = copy.deepcopy(model)
            f_optimizer = self.get_optim(self.f_optim)(fold_model.parameters(), lr=self.f_lr,
                                                       weight_decay=self.f_weight_decay)
            with trange(self.f_epoch) as t:
                for epoch in t:
                    t.set_description('Fold %d, finetuning' % (fold+1))
                    self.finetune(fold_model, f_optimizer, train_loader)
                    val_loss = self.eval_loss(fold_model, val_loader)
                    test_score = self.eval_metric(fold_model, test_loader)
                    val_losses.append(val_loss)
                    test_scores.append(test_score)

                    t.set_postfix(val_loss='{:.4f}'.format(val_loss), 
                                  acc='{:.4f}'.format(test_score))
        

        val_losses, test_scores = torch.tensor(val_losses), torch.tensor(test_scores)
        val_losses = val_losses.view(self.n_folds, self.f_epoch)
        test_scores = test_scores.view(self.n_folds, self.f_epoch)

        if self.epoch_select == 'test_max':
            _, selection =  test_scores.mean(dim=0).max(dim=0)
            selection = selection.repeat(self.n_folds)
        elif self.epoch_select == 'test_min':
            _, selection =  test_scores.mean(dim=0).min(dim=0)
            selection = selection.repeat(self.n_folds)
        else:
            _, selection =  val_losses.min(dim=1)

        test_scores = test_scores[torch.arange(self.n_folds, dtype=torch.long), selection]
        test_acc_mean = test_scores.mean().item()
        test_acc_std = test_scores.std().item() 
        
        return test_acc_mean, test_acc_std
    
    
    def grid_search(self, learning_model, encoder, pred_head=None, fold_seed=12345,
                    p_lr_lst=[0.1,0.01,0.001,0.0001], p_epoch_lst=[20,40,60,80,100]):
        
        r"""Perform grid search on learning rate and epochs in pretraining.
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive) 
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            pred_head (torch.nn.Module, optional): Prediction head. If None, will use linear 
                projection. (default: :obj:`None`)
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
                acc_m, acc_sd = self.evaluate(learning_model, encoder, pred_head, fold_seed)
                acc_m_lst.append(acc_m)
                acc_sd_lst.append(acc_sd)
                paras.append((p_lr, p_epoch))
        idx = np.argmax(acc_m_lst)
        print('Best paras: %d epoch, lr=%f, acc=%.4f' %(
            paras[idx][1], paras[idx][0], acc_m_lst[idx]))
        
        return acc_m_lst[idx], acc_sd_lst[idx], paras[idx]

    
    def finetune(self, model, optimizer, loader):
        
        model.train()
        for data in loader:
            optimizer.zero_grad()
            data = data.to(self.device)
            out = model(data)
            loss = self.loss(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
                
    
    def eval_loss(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()

        loss = 0
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = model(data)
            loss += self.loss(pred, data.y.view(-1), reduction='sum').item()
            
        return loss / len(loader.dataset)
    
    
    def eval_acc(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()

        correct = 0
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = model(data).max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            
        return correct / len(loader.dataset)
    
        
    def get_optim(self, optim):
        
        if callable(optim):
            return optim
        
        optims = {'Adam': torch.optim.Adam}
        
        return optims[optim]
    
    def eval_metric(self, model, loader, eval_mode=True):
        
        if self.metric == 'acc':
            return self.eval_acc(model, loader, eval_mode)
        
    
    
def k_fold(n_folds, dataset, batch_size, label_rate=1, val=False, seed=12345):
    
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if val:
        val_indices = [test_indices[i - 1] for i in range(n_folds)]
    else:
        val_indices = [test_indices[i] for i in range(n_folds)]

    if label_rate < 1:
        label_skf = StratifiedKFold(int(1.0/label_rate), shuffle=True, random_state=seed)
        for i in range(n_folds):
            train_mask = torch.ones(len(dataset), dtype=torch.uint8)
            train_mask[test_indices[i].long()] = 0
            train_mask[val_indices[i].long()] = 0
            idx_train = train_mask.nonzero(as_tuple=False).view(-1)
            for _, idx in label_skf.split(torch.zeros(idx_train.size()[0]), 
                                          dataset.data.y[idx_train]):
                idx_train = idx_train[idx]
                break

            train_indices.append(idx_train)
    else:
        for i in range(n_folds):
            train_mask = torch.ones(len(dataset), dtype=torch.uint8)
            train_mask[test_indices[i].long()] = 0
            train_mask[val_indices[i].long()] = 0
            idx_train = train_mask.nonzero(as_tuple=False).view(-1)
            train_indices.append(idx_train)
            
    for i in range(n_folds):
        if batch_size is None:
            batch_size = len()
        train_loader = DataLoader(dataset[train_indices[i].long()], batch_size, shuffle=True)
        test_loader = DataLoader(dataset[test_indices[i].long()], batch_size, shuffle=False)
        val_loader = DataLoader(dataset[val_indices[i].long()], batch_size, shuffle=False)

        yield i, train_loader, test_loader, val_loader
