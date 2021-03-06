import time
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad

def run(train_dataset, val_dataset, model, epochs, batch_size, lr, lr_decay_factor, lr_decay_step_size, weight_decay, save_dir, 
        energy_and_force, num_atom, p):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.L1Loss(reduction='none')

    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)    
    best_val = float('inf')
    epoch_best_val = 0
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, 'model.ckpt')    

    
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        t_start = time.perf_counter()

        # model.train()
        # losses = []
        # for batch_data in train_loader:
        #     optimizer.zero_grad()
        #     batch_data = batch_data.to(device)
        #     out = model(batch_data)
        #     if energy_and_force:
        #         force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
        #         e_loss = loss_func(out, batch_data.y.unsqueeze(1))
        #         f_loss = loss_func(force, batch_data.force)
        #         loss = e_loss.sum() + p/(3*num_atom) * f_loss.sum()
        #     else:
        #         loss = loss_func(out, batch_data.y.unsqueeze(1))
        #     loss.backward()
        #     optimizer.step()
        #     losses.append(loss)

        train_loss = train(model, optimizer, train_loader, energy_and_force, num_atom, p, loss_func, device)/len(train_dataset) 

        # model.eval()
        # losses = []
        # for batch_data in val_loader:
        #     optimizer.zero_grad()
        #     batch_data = batch_data.to(device)
        #     out = model(batch_data)
        #     if energy_and_force:
        #         force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
        #         e_loss = loss_func(out, batch_data.y.unsqueeze(1))
        #         f_loss = loss_func(force, batch_data.force)
        #         loss = e_loss.sum() + p/(3*num_atom) * f_loss.sum()
        #     else:
        #         loss = loss_func(out, batch_data.y.unsqueeze(1))
        #     losses.append(loss)
        val_loss = val(model, optimizer, val_loader, energy_and_force, num_atom, p, loss_func, device)/len(val_dataset)

        
        if val_loss < best_val:
            epoch_best_val = epoch
            best_val = val_loss
            torch.save(model.state_dict(), save_dir)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        
        print('Epoch: {:03d}, Training Loss: {:.4f}, Val Loss: {:.4f}, Epoch_best_val: {:03d}, Duration: {:.2f}'.format(
            epoch, train_loss,  val_loss, epoch_best_val, t_end - t_start))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

def train(model, optimizer, train_loader, energy_and_force, num_atom, p, loss_func, device):
    model.train()
    losses = []
    for batch_data in train_loader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        # print('out',out)
        # print('y',batch_data.y.unsqueeze(1))
        if energy_and_force:
            force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
            e_loss = loss_func(out, batch_data.y.unsqueeze(1))
            f_loss = loss_func(force, batch_data.force)
            loss = e_loss.sum() + p/(3*num_atom) * f_loss.sum()
        else:
            loss = loss_func(out, batch_data.y.unsqueeze(1)).sum()
            # print('batch_loss',loss)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return sum(losses).item()

def val(model, optimizer, val_loader, energy_and_force, num_atom, p, loss_func, device):
    model.eval()
    losses = []
    for batch_data in val_loader:
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
        losses.append(loss)
    return sum(losses).item()