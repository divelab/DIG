import time
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from metric import compute_cla_metric, compute_reg_metric
import numpy as np

from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### This is run function for classification tasks
def run_classification(train_dataset, val_dataset, test_dataset, model, num_tasks, epochs, batch_size, vt_batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, early_stopping, metric, log_dir, save_dir, evaluate):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, vt_batch_size, shuffle=False)
    
    best_val_metric = 0
    val_loss_history = []
    
    epoch_bvl = 0
    
    writer = None
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = os.path.join(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_dir = os.path.join(save_dir, 'params.ckpt') 
    
   
    for epoch in range(1, epochs + 1):
        ### synchronizing helps to record more accurate running time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        
        t_start = time.perf_counter()
        
        total_train_loss = train_classification(model, optimizer, train_loader, num_tasks, ce_loss, device) 
        
        val_prc_results, val_roc_results, total_val_loss = val_classification(model, val_loader, num_tasks, ce_loss, device)

        
        ### All loss we consider is per-sample loss
        train_loss_per_smaple = total_train_loss/len(train_dataset)
        val_loss_per_smaple = total_val_loss/len(val_dataset)
        
        
        if writer is not None:
            writer.add_scalar('train_loss_per_sample', train_loss_per_smaple, epoch)
            writer.add_scalar('val_loss_per_smaple', val_loss_per_smaple, epoch)
            if metric == "prc":
                writer.add_scalar('Val PRC', np.mean(val_prc_results), epoch)
            elif metric == "roc":
                writer.add_scalar('Val ROC', np.mean(val_roc_results), epoch)

        ### One possible way to selection model: do testing when val metric is best
        if metric == "prc":
            if np.mean(val_prc_results) > best_val_metric:
                epoch_bvl = epoch
                best_val_metric = np.mean(val_prc_results)
                torch.save(model.state_dict(), save_dir)
        elif metric == "roc":
            if np.mean(val_roc_results) > best_val_metric:
                epoch_bvl = epoch
                best_val_metric = np.mean(val_roc_results)
                torch.save(model.state_dict(), save_dir)
        else:
            print("Metric is not consistent with task type!!!")

        ### One possible way to stop training    
        val_loss_history.append(val_loss_per_smaple)
        if early_stopping > 0 and epoch > epochs // 2 and epoch > early_stopping:
            tmp = torch.tensor(val_loss_history[-(early_stopping + 1):-1])
            if val_loss_per_smaple > tmp.mean().item():
                break
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        

        print('Epoch: {:03d}, Training Loss: {:.6f}, Val Loss: {:.6f}, Val PRC (avg over multitasks): {:.4f}, Val ROC (avg over multitasks): {:.4f}, Duration: {:.2f}'.format(
            epoch, train_loss_per_smaple, val_loss_per_smaple, np.mean(val_prc_results), np.mean(val_roc_results), t_end - t_start))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    if writer is not None:
        writer.close()
        
    print('======================')    
    print('Stop training at epoch:', epoch, '; Best val metric before this epoch is:', best_val_metric, '; Best val metric achieves at epoch:', epoch_bvl)       
    print('======================') 
    
    if evaluate:
        print('Loading trained model and testing...')
        model.load_state_dict(torch.load(save_dir))
        test_prc_results, test_roc_results = test_classification(model, test_loader, num_tasks, device)


        print('======================')        
        print('Epoch: {:03d}, Test PRC (avg over multitasks): {:.4f}, Test ROC (avg over multitasks): {:.4f}'.format(epoch_bvl, np.mean(test_prc_results), np.mean(test_roc_results)))
        print('======================')
        print('Test PRC for all tasks:', test_prc_results)
        print('Test ROC for all tasks:', test_roc_results)
        print('======================')


    
def train_classification(model, optimizer, train_loader, num_tasks, ce_loss, device):
    model.train()

    losses = []
    for batch_data in train_loader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in batch_data.y.cpu()]) # Skip those without targets (in PCBA, MUV, Tox21, ToxCast)
        mask = mask.to(device)
        target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch_data.y.cpu()])
        target = target.to(device)
        loss = ce_loss(out, target) * mask
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return sum(losses).item()



def val_classification(model, val_loader, num_tasks, ce_loss, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    losses = []
    for batch_data in val_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in batch_data.y.cpu()])
        mask = mask.to(device)
        target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch_data.y.cpu()])
        target = target.to(device)
        loss = ce_loss(out, target) * mask
        loss = loss.sum()
        losses.append(loss)
        pred = torch.sigmoid(out) ### prediction real number between (0,1)
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
    
    prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    
    return prc_results, roc_results, sum(losses).item()



def test_classification(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = torch.sigmoid(out) ### prediction real number between (0,1)
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
    prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    
    return prc_results, roc_results



### This is run function for regression tasks
def run_regression(train_dataset, val_dataset, test_dataset, model, num_tasks, epochs, batch_size, vt_batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, early_stopping, metric, log_dir, save_dir, evaluate):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = torch.nn.MSELoss(reduction='none')

    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, vt_batch_size, shuffle=False)
    
    best_val_metric = float('inf')
    val_loss_history = []

    epoch_bvl = 0
    
    writer = None
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = os.path.join(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, 'params.ckpt')  
    
    for epoch in range(1, epochs + 1):
        ### synchronizing helps to record more accurate running time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True) # Use drop_last to avoid get a 1 sample batch, in which case BatchNorm does not work.

        t_start = time.perf_counter()
        total_train_loss = train_regression(model, optimizer, train_loader, num_tasks, mse_loss, device) 
        val_mae_results, val_rmse_results, total_val_loss = val_regression(model, val_loader, num_tasks, mse_loss, device)
        
        
        ### All loss we consider is per-sample loss
        train_loss_per_smaple = total_train_loss/len(train_dataset)
        val_loss_per_smaple = total_val_loss/len(val_dataset)
        
        if writer is not None:
            writer.add_scalar('train_loss_per_sample', train_loss_per_smaple, epoch)
            writer.add_scalar('val_loss_per_smaple', val_loss_per_smaple, epoch)
            if metric == "mae":
                writer.add_scalar('Val MAE', np.mean(val_mae_results), epoch)
            elif metric == "rmse":
                writer.add_scalar('Val RMSE', np.mean(val_rmse_results), epoch)
        
        ### One possible way to selection model: do testing when val metric is best
        if metric == "mae":
            if np.mean(val_mae_results) < best_val_metric:
                epoch_bvl = epoch
                best_val_metric = np.mean(val_mae_results)

                torch.save(model.state_dict(), save_dir)
        elif metric == "rmse":
            if np.mean(val_rmse_results) < best_val_metric:
                epoch_bvl = epoch
                best_val_metric = np.mean(val_rmse_results)
                torch.save(model.state_dict(), save_dir)
        else:
            print("Metric is not consistent with task type!!!")
            
        ### One possible way to stop training    
        val_loss_history.append(val_loss_per_smaple)
        if early_stopping > 0 and epoch > epochs // 2 and epoch > early_stopping:
            tmp = torch.tensor(val_loss_history[-(early_stopping + 1):-1])
            if val_loss_per_smaple > tmp.mean().item():
                break
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        
        print('Epoch: {:03d}, Training Loss: {:.6f}, Val Loss: {:.6f}, Val MAE (avg over multitasks): {:.4f}, Val RMSE (avg over multitasks): {:.4f}, Duration: {:.2f}'.format(
            epoch, train_loss_per_smaple, val_loss_per_smaple, np.mean(val_mae_results), np.mean(val_rmse_results), t_end - t_start))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    
        
    if writer is not None:
        writer.close() 
    
    print('======================')    
    print('Stop training at epoch:', epoch, '; Best val metric before this epoch is:', best_val_metric, '; Best val metric achieves at epoch:', epoch_bvl)       
    print('======================') 
    
    if evaluate:
        print('Loading trained model and testing...')
        model.load_state_dict(torch.load(save_dir))
        test_mae_results, test_rmse_results = test_regression(model, test_loader, num_tasks, device)

        print('======================')        
        print('Epoch: {:03d}, Test MAE (avg over multitasks): {:.4f}, Test RMSE (avg over multitasks): {:.4f}'.format(epoch_bvl, np.mean(test_mae_results), np.mean(test_rmse_results)))
        print('======================')
        print('Test MAE for all tasks:', test_mae_results)
        print('Test RMSE for all tasks:', test_rmse_results)
        print('======================')
    
    

def train_regression(model, optimizer, train_loader, num_tasks, mse_loss, device):
    model.train()

    losses = []
    for batch_data in train_loader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        loss = mse_loss(out, batch_data.y)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return sum(losses).item()
              

    
def val_regression(model, val_loader, num_tasks, mse_loss, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    losses = []
    for batch_data in val_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        loss = mse_loss(out, batch_data.y)
        loss = loss.sum()
        losses.append(loss)
        pred = out 
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
      
    mae_results, rmse_results = compute_reg_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    
    return mae_results, rmse_results, sum(losses).item()



def test_regression(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = out
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)

    mae_results, rmse_results = compute_reg_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)

    return mae_results, rmse_results