import torch
from torch.optim import Adam
import time
from tqdm import tqdm
import os

from dig.ggraph.method import Generator
from dig.ggraph.utils import gen_mol_from_one_shot_tensor
from .energy_func import EnergyFunc
from .util import rescale_adj, requires_grad, clip_grad


class GraphEBM(Generator):
    def __init__(self, n_atom, n_atom_type, n_edge_type, hidden, device=None):
        super(GraphEBM, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.energy_function = EnergyFunc(n_atom_type, hidden, n_edge_type).to(self.device)
        self.n_atom = n_atom
        self.n_atom_type = n_atom_type
        self.n_edge_type = n_edge_type
    
    
    def train_rand_gen(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size, clamp, alpha, save_interval, save_dir):
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for epoch in range(max_epochs):
            t_start = time.time()
            losses_reg = []
            losses_en = []
            losses = []
            for i, batch in enumerate(tqdm(loader)):
                ### Dequantization
                pos_x = batch.x.to(self.device).to(dtype=torch.float32)
                pos_x += c * torch.rand_like(pos_x, device=self.device)  
                pos_adj = batch.adj.to(self.device).to(dtype=torch.float32)
                pos_adj += c * torch.rand_like(pos_adj, device=self.device)  


                ### Langevin dynamics
                neg_x = torch.rand_like(pos_x, device=self.device) * (1 + c) 
                neg_adj = torch.rand_like(pos_adj, device=self.device) 

                pos_adj = rescale_adj(pos_adj)
                neg_x.requires_grad = True
                neg_adj.requires_grad = True



                requires_grad(parameters, False)
                self.energy_function.eval()



                noise_x = torch.randn_like(neg_x, device=self.device)
                noise_adj = torch.randn_like(neg_adj, device=self.device)
                for k in range(ld_step):

                    noise_x.normal_(0, ld_noise)
                    noise_adj.normal_(0, ld_noise)
                    neg_x.data.add_(noise_x.data)
                    neg_adj.data.add_(noise_adj.data)

                    neg_out = self.energy_function(neg_adj, neg_x)
                    neg_out.sum().backward()
                    if clamp:
                        neg_x.grad.data.clamp_(-0.01, 0.01)
                        neg_adj.grad.data.clamp_(-0.01, 0.01)


                    neg_x.data.add_(neg_x.grad.data, alpha=ld_step_size)
                    neg_adj.data.add_(neg_adj.grad.data, alpha=ld_step_size)

                    neg_x.grad.detach_()
                    neg_x.grad.zero_()
                    neg_adj.grad.detach_()
                    neg_adj.grad.zero_()

                    neg_x.data.clamp_(0, 1 + c)
                    neg_adj.data.clamp_(0, 1)

                ### Training by backprop
                neg_x = neg_x.detach()
                neg_adj = neg_adj.detach()
                requires_grad(parameters, True)
                self.energy_function.train()

                self.energy_function.zero_grad()

                pos_out = self.energy_function(pos_adj, pos_x)
                neg_out = self.energy_function(neg_adj, neg_x)

                loss_reg = (pos_out ** 2 + neg_out ** 2)  # energy magnitudes regularizer
                loss_en = pos_out - neg_out  # loss for shaping energy function
                loss = loss_en + alpha * loss_reg
                loss = loss.mean()
                loss.backward()
                clip_grad(parameters, optimizer)
                optimizer.step()


                losses_reg.append(loss_reg.mean())
                losses_en.append(loss_en.mean())
                losses.append(loss)
            
            
            t_end = time.time()

            ### Save checkpoints
            if (epoch+1) % save_interval == 0:
                torch.save(self.energy_function.state_dict(), os.path.join(save_dir, 'epoch_{}.pt'.format(epoch + 1)))
                print('Saving checkpoint at epoch ', epoch+1)
                print('==========================================')
            print('Epoch: {:03d}, Loss: {:.6f}, Energy Loss: {:.6f}, Regularizer Loss: {:.6f}, Sec/Epoch: {:.2f}'.format(epoch+1, (sum(losses)/len(losses)).item(), (sum(losses_en)/len(losses_en)).item(), (sum(losses_reg)/len(losses_reg)).item(), t_end-t_start))
            print('==========================================')
    
    
    def run_rand_gen(self, checkpoint_path, n_samples, c, ld_step, ld_noise, ld_step_size, clamp, atomic_num_list):
        print("Loading paramaters from {}".format(checkpoint_path))
        self.energy_function.load_state_dict(torch.load(checkpoint_path))
        parameters =  self.energy_function.parameters()
        
        ### Initialization
        print("Initializing samples...")
        gen_x = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)
        
        gen_x.requires_grad = True
        gen_adj.requires_grad = True
        requires_grad(parameters, False)
        self.energy_function.eval()
        
        noise_x = torch.randn_like(gen_x, device=self.device)
        noise_adj = torch.randn_like(gen_adj, device=self.device)
        
        ### Langevin dynamics
        print("Generating samples...")
        for k in range(ld_step):
            noise_x.normal_(0, ld_noise)
            noise_adj.normal_(0, ld_noise)
            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)


            gen_out = self.energy_function(gen_adj, gen_x)
            gen_out.sum().backward()
            if clamp:
                gen_x.grad.data.clamp_(-0.01, 0.01)
                gen_adj.grad.data.clamp_(-0.01, 0.01)


            gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
            gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)

            gen_x.grad.detach_()
            gen_x.grad.zero_()
            gen_adj.grad.detach_()
            gen_adj.grad.zero_()

            gen_x.data.clamp_(0, 1 + c)
            gen_adj.data.clamp_(0, 1)
            
        gen_x = gen_x.detach()
        gen_adj = gen_adj.detach()
        gen_adj = (gen_adj + gen_adj.permute(0, 1, 3, 2)) / 2
        
        gen_mols = gen_mol_from_one_shot_tensor(gen_adj, gen_x, atomic_num_list, correct_validity=True)
        
        return gen_mols


        

    def train_prop_optim(self, *args, **kwargs):
        raise NotImplementedError("The function train_prop_optim is not implemented!")
    
    def run_prop_optim(self, *args, **kwargs):
        raise NotImplementedError("The function run_prop_optim is not implemented!")
    
    def train_cons_optim(self, loader, *args, **kwargs):
        raise NotImplementedError("The function train_cons_optim is not implemented!")
    
    def run_cons_optim(self, loader, *args, **kwargs):
        raise NotImplementedError("The function run_cons_optim is not implemented!")