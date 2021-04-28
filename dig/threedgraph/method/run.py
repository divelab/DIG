
import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
# import sys
# sys.path.append('..')
# from utils import compute_mae

class run():
    r"""
    The base script for running different 3DGN methods.
    """

    def run(self, train_dataset, val_dataset, test_dataset, save_dir, log_dir, model, epochs, batch_size, lr, lr_decay_factor, lr_decay_step_size, weight_decay, 
        energy_and_force, num_atom, p):
        r"""
        The run script for traning and validation.
        
        Args:
            train_dataset (str): The path of the training set.
            val_dataset (str): The path of the validation set.
            test_dataset (str): The path of the test set.
            save_dir (str): The path to save trained models.
            log_dir (str): The path to save log files.
            model (str): Which 3DGN model to use. Should be one of the schnet, dimenetpp, and spherenet.
            epochs (int): Number of total training epochs.
            batch_size (int, optinal): Number of samples in each minibatch. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0001`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.1`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`8`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0.9999`)
            energy_and_force (bool, optional): If set to :obj:`True`, will preddict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            num_atom (int, optinal): Maximal number of atoms for a training sample. A sample with larger value is skipped and not used. (default: :obj:`2000`)
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
        
        """        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_func = torch.nn.L1Loss(reduction='none')
        metric_func = torch.nn.L1Loss(reduction='none') #compute_mae #the metrics can be different with the loss function

        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        best_val = float('inf')
        epoch_best_val = 0
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, 'model.ckpt') 
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = os.path.join(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        
        for epoch in range(1, epochs + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            t_start = time.perf_counter()

            train_loss = self.train(model, optimizer, train_loader, energy_and_force, num_atom, p, loss_func, device)/len(train_dataset) 
            val_metric, val_loss = self.val(model, val_loader, energy_and_force, num_atom, p, loss_func, metric_func, device)
            val_loss = val_loss/len(val_dataset)

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_metric', val_metric, epoch)
            
            if val_metric < best_val:
                epoch_best_val = epoch
                best_val = val_metric
                torch.save(model.state_dict(), save_dir)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            
            print('Epoch: {:03d}, Training Loss: {:.4f}, Val Loss: {:.4f}, Epoch_best_val: {:03d}, Duration: {:.2f}'.format(
                epoch, train_loss,  val_loss, epoch_best_val, t_end - t_start))

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
        writer.close()

    def train(self, model, optimizer, train_loader, energy_and_force, num_atom, p, loss_func, device):
        r"""
        The script for training.
        
        Args:
            model (str): Which 3DGN model to use. Should be one of the schnet, dimenetpp, and spherenet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will preddict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            num_atom (int, optinal): Maximal number of atoms for a training sample. A sample with larger value is skipped and not used. (default: :obj:`2000`)
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function, optional): The used loss funtion for training. (default: MSE)
            device (torch.device, optional): The device where the model is deployed.
        
        """   
        model.train()
        losses = []
        for batch_data in train_loader:
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                f_loss = loss_func(force, batch_data.force)
                loss = e_loss.sum() + p/(3*num_atom) * f_loss.sum()
            else:
                loss = loss_func(out, batch_data.y.unsqueeze(1)).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss)
        return sum(losses).item()

    def val(self, model, val_loader, energy_and_force, num_atom, p, loss_func, metric_func, device):
        r"""
        The script for validation.
        
        Args:
            model (str): Which 3DGN model to use. Should be one of the schnet, dimenetpp, and spherenet.
            val_loader (Dataloader): Dataloader for validation.
            energy_and_force (bool, optional): If set to :obj:`True`, will preddict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            num_atom (int, optinal): Maximal number of atoms for a training sample. A sample with larger value is skipped and not used. (default: :obj:`2000`)
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function, optional): The used loss funtion in training. (default: MSE)
            metric_func (function, optional): The used funtion for evaluation. (default: MSE)
            device (torch.device, optional): The device where the model is deployed.
        
        """   
        model.eval()
        losses = torch.Tensor([0.0]).to(device)
        if energy_and_force:
            preds_energy = torch.Tensor([]).to(device)
            preds_force = torch.Tensor([]).to(device)
            targets_energy = torch.Tensor([]).to(device)
            targets_force = torch.Tensor([]).to(device)
        else:
            preds = torch.Tensor([]).to(device)
            targets = torch.Tensor([]).to(device)
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                preds_force = torch.cat([preds_force,force.detach_()], dim=0)
                preds_energy = torch.cat([preds_energy, out.detach_()], dim=0)
                targets_force = torch.cat([targets_force,batch_data.force], dim=0)
                targets_energy = torch.cat([targets_energy, batch_data.y.unsqueeze(1)], dim=0)
                e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                f_loss = loss_func(force, batch_data.force)
                loss = e_loss.sum() + p/(3*num_atom) * f_loss.sum()
            else:
                preds = torch.cat([preds, out.detach_()], dim=0)
                targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)
                loss = loss_func(out, batch_data.y.unsqueeze(1)).sum()
            losses += loss.sum().item()
        if energy_and_force:
            energy_metric = metric_func(targets_energy.cpu().detach().numpy(), preds_energy.cpu().detach().numpy())
            force_metric = metric_func(targets_force.cpu().detach().numpy(), preds_force.cpu().detach().numpy(), num_tasks=num_atom*3)
            return np.mean(energy_metric) + p * np.mean(force_metric), losses[0]
        return np.mean(metric_func(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())), losses[0]
