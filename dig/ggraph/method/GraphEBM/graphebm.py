import time
import os
import copy

import torch
from torch.optim import Adam
from tqdm import tqdm
from rdkit import Chem



from dig.ggraph.method import Generator
from dig.ggraph.utils import gen_mol_from_one_shot_tensor
from dig.ggraph.utils import qed, calculate_min_plogp, reward_target_molecule_similarity
from .energy_func import EnergyFunc
from .util import rescale_adj, requires_grad, clip_grad


class GraphEBM(Generator):
    r"""
        The method class for GraphEBM algorithm proposed in the paper `GraphEBM: Molecular Graph Generation with Energy-Based Models <https://arxiv.org/abs/2102.00546>`_. This class provides interfaces for running random generation, goal-directed generation (including property
        optimization and constrained optimization), and compositional generation with GraphEBM algorithm. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphEBM>`_ for usage examples.
        
        Args:
            n_atom (int): Maximum number of atoms.
            n_atom_type (int): Number of possible atom types.
            n_edge_type (int): Number of possible bond types.
            hidden (int): Hidden dimensions.
            device (torch.device, optional): The device where the model is deployed.

    """
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
        r"""
            Running training for random generation task.

            Args:
                loader: The data loader for loading training samples. It is supposed to use dig.ggraph.dataset.QM9/ZINC250k
                    as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                lr (float): The learning rate for training.
                wd (float): The weight decay factor for training.
                max_epochs (int): The maximum number of training epochs.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                alpha (float): The weight coefficient for loss function.
                save_interval (int): The frequency to save the model parameters to .pt files,
                    *e.g.*, if save_interval=2, the model parameters will be saved for every 2 training epochs.
                save_dir (str): the directory to save the model parameters.
        """
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for epoch in range(max_epochs):
            t_start = time.time()
            losses_reg = []
            losses_en = []
            losses = []
            for _, batch in enumerate(tqdm(loader)):
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
                for _ in range(ld_step):

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
                clip_grad(optimizer)
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
        r"""
            Running graph generation for random generation task.

            Args:
                checkpoint_path (str): The path of the trained model, *i.e.*, the .pt file.
                n_samples (int): the number of molecules to generate.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                atomic_num_list (list): The list used to indicate atom types. 
            
            :rtype:
                gen_mols (list): A list of generated molecules represented by rdkit Chem.Mol objects;
                
        """
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
        for _ in range(ld_step):
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


    def train_goal_directed(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size, clamp, alpha, save_interval, save_dir):
        r"""
            Running training for goal-directed generation task.

            Args:
                loader: The data loader for loading training samples. It is supposed to use dig.ggraph.dataset.QM9/ZINC250k
                    as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                lr (float): The learning rate for training.
                wd (float): The weight decay factor for training.
                max_epochs (int): The maximum number of training epochs.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                alpha (float): The weight coefficient for loss function.
                save_interval (int): The frequency to save the model parameters to .pt files,
                    *e.g.*, if save_interval=2, the model parameters will be saved for every 2 training epochs.
                save_dir (str): the directory to save the model parameters.
        """
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for epoch in range(max_epochs):
            t_start = time.time()
            losses_reg = []
            losses_en = []
            losses = []
            for _, batch in enumerate(tqdm(loader)):
                ### Dequantization
                pos_x = batch.x.to(self.device).to(dtype=torch.float32)
                pos_x += c * torch.rand_like(pos_x, device=self.device)  
                pos_adj = batch.adj.to(self.device).to(dtype=torch.float32)
                pos_adj += c * torch.rand_like(pos_adj, device=self.device) 
                
                pos_y = batch.y.to(self.device)


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
                for _ in range(ld_step):

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
                loss_en = (1 + torch.exp(pos_y)) * pos_out - neg_out  # loss for shaping energy function
                loss = loss_en + alpha * loss_reg
                loss = loss.mean()
                loss.backward()
                clip_grad(optimizer)
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
    

    def run_prop_opt(self, checkpoint_path, initialization_loader, c, ld_step, ld_noise, ld_step_size, clamp, atomic_num_list, train_smiles):
        r"""
            Running graph generation for goal-directed generation task: property optimization.

            Args:
                checkpoint_path (str): The path of the trained model, *i.e.*, the .pt file.
                initialization_loader: The data loader for loading samples to initialize the Langevin dynamics. It is supposed to use dig.ggraph.dataset.QM9/ZINC250k as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                atomic_num_list (list): The list used to indicate atom types. 
                train_smiles (list): A list of smiles string corresponding to training samples.
            
            :rtype:
                save_mols_list (list), prop_list (list): save_mols_list is a list of generated molecules with high QED scores represented by rdkit Chem.Mol objects; prop_list is a list of the corresponding QED scores.
                
        """
        print("Loading paramaters from {}".format(checkpoint_path))
        self.energy_function.load_state_dict(torch.load(checkpoint_path))
        parameters =  self.energy_function.parameters()
        
        save_mols_list = []
        prop_list = []
        
        for _, batch in enumerate(tqdm(initialization_loader)): 
            ### Initialization
            gen_x = batch.x.to(self.device).to(dtype=torch.float32)
            gen_adj = batch.adj.to(self.device).to(dtype=torch.float32)

            gen_x.requires_grad = True
            gen_adj.requires_grad = True
            requires_grad(parameters, False)
            self.energy_function.eval()

            noise_x = torch.randn_like(gen_x, device=self.device)
            noise_adj = torch.randn_like(gen_adj, device=self.device)

            ### Langevin dynamics
            for _ in range(ld_step):
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
                
                gen_x_t = copy.deepcopy(gen_x)
                gen_adj_t = copy.deepcopy(gen_adj)
                gen_adj_t = (gen_adj_t + gen_adj_t.permute(0, 1, 3, 2)) / 2  
                
                gen_mols = gen_mol_from_one_shot_tensor(gen_adj_t, gen_x_t, atomic_num_list, correct_validity=True)
                gen_smiles = [Chem.MolToSmiles(mol) for mol in gen_mols]

                for mol_idx in range(len(gen_smiles)):
                    if gen_mols[mol_idx] is not None:
                        tmp_mol = gen_mols[mol_idx]
                        tmp_smiles = gen_smiles[mol_idx]
                        if tmp_smiles not in train_smiles:
                            tmp_qed = qed(tmp_mol)
                            if tmp_qed > 0.930:
                                save_mols_list.append(tmp_mol)
                                prop_list.append(tmp_qed)
        return save_mols_list, prop_list
    
    
    def run_const_prop_opt(self, checkpoint_path, initialization_loader, c, ld_step, ld_noise, ld_step_size, clamp, atomic_num_list, train_smiles):
        r"""
            Running graph generation for goal-directed generation task: constrained property optimization.

            Args:
                checkpoint_path (str): The path of the trained model, *i.e.*, the .pt file.
                initialization_loader: The data loader for loading samples to initialize the Langevin dynamics. It is supposed to use dig.ggraph.dataset.QM9/ZINC250k as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                atomic_num_list (list): The list used to indicate atom types.
                train_smiles (list): A list of smiles string corresponding to training samples.
            
            :rtype:
                mols_0_list (list), mols_2_list (list), mols_4_list (list), mols_6_list (list), imp_0_list (list), imp_2_list (list), imp_4_list (list), imp_4_list (list): They are lists of optimized molecules (represented by rdkit Chem.Mol objects) and the corresponding improvements under the threshold 0.0, 0.2, 0.4, 0.6, respectively.   
        """
        print("Loading paramaters from {}".format(checkpoint_path))
        self.energy_function.load_state_dict(torch.load(checkpoint_path))
        parameters =  self.energy_function.parameters()
        
        mols_0_list = [None]*800
        mols_2_list = [None]*800
        mols_4_list = [None]*800
        mols_6_list = [None]*800
        
        imp_0_list = [0]*800
        imp_2_list = [0]*800
        imp_4_list = [0]*800
        imp_6_list = [0]*800
        
        for i, batch in enumerate(tqdm(initialization_loader)): 
            ### Initialization
            gen_x = batch.x.to(self.device).to(dtype=torch.float32)
            gen_adj = batch.adj.to(self.device).to(dtype=torch.float32)
            
            ori_mols = gen_mol_from_one_shot_tensor(gen_adj, gen_x, atomic_num_list, correct_validity=True)
            ori_smiles = [Chem.MolToSmiles(mol) for mol in ori_mols]

            gen_x.requires_grad = True
            gen_adj.requires_grad = True
            requires_grad(parameters, False)
            self.energy_function.eval()

            noise_x = torch.randn_like(gen_x, device=self.device)
            noise_adj = torch.randn_like(gen_adj, device=self.device)

            ### Langevin dynamics
            for k in range(ld_step):
                noise_x.normal_(0, ld_noise)
                noise_adj.normal_(0, ld_noise)
                gen_x.data.add_(noise_x.data)
                gen_adj.data.add_(noise_adj.data)


                gen_out = self.energy_function(gen_adj, gen_x)
                gen_out.sum().backward()
                if clamp:
                    gen_x.grad.data.clamp_(-0.1, 0.1)
                    gen_adj.grad.data.clamp_(-0.1, 0.1)


                gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
                gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)

                gen_x.grad.detach_()
                gen_x.grad.zero_()
                gen_adj.grad.detach_()
                gen_adj.grad.zero_()

                gen_x.data.clamp_(0, 1 + c)
                gen_adj.data.clamp_(0, 1)
                
                gen_x_t = copy.deepcopy(gen_x)
                gen_adj_t = copy.deepcopy(gen_adj)
                gen_adj_t = (gen_adj_t + gen_adj_t.permute(0, 1, 3, 2)) / 2  
                
                gen_mols = gen_mol_from_one_shot_tensor(gen_adj_t, gen_x_t, atomic_num_list, correct_validity=True)
                gen_smiles = [Chem.MolToSmiles(mol) for mol in gen_mols]

                for mol_idx in range(len(gen_smiles)):
                    if gen_mols[mol_idx] is not None:
                        tmp_mol = gen_mols[mol_idx]
                        ori_mol = ori_mols[mol_idx]
                        imp_p = calculate_min_plogp(tmp_mol) - calculate_min_plogp(ori_mol)
                        current_sim = reward_target_molecule_similarity(tmp_mol, ori_mol)
                        if current_sim >= 0.:
                            if imp_p > imp_0_list[mol_idx]:
                                mols_0_list[mol_idx] = tmp_mol
                        if current_sim >= 0.2:
                            if imp_p > imp_2_list[mol_idx]:
                                mols_2_list[mol_idx] = tmp_mol
                        if current_sim >= 0.4:
                            if imp_p > imp_4_list[mol_idx]:
                                mols_4_list[mol_idx] = tmp_mol
                        if current_sim >= 0.6:
                            if imp_p > imp_6_list[mol_idx]:
                                mols_6_list[mol_idx] = tmp_mol
                                
        return mols_0_list, mols_2_list, mols_4_list, mols_6_list, imp_0_list, imp_2_list, imp_4_list, imp_4_list
    
    
    def run_comp_gen(self, checkpoint_path_qed, checkpoint_path_plogp, n_samples, c, ld_step, ld_noise, ld_step_size, clamp, atomic_num_list):
        r"""
            Running graph generation for compositional generation task.

            Args:
                checkpoint_path_qed (str): The path of the model trained on QED property, *i.e.*, the .pt file.
                checkpoint_path_plogp (str): The path of the model trained on plogp property, *i.e.*, the .pt file.
                n_samples (int): the number of molecules to generate.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                atomic_num_list (list): The list used to indicate atom types.
            
            :rtype:
                gen_mols (list): A list of generated molecules represented by rdkit Chem.Mol objects;
        """
        model_qed = self.energy_function
        model_plogp = copy.deepcopy(self.energy_function)
        print("Loading paramaters from {}".format(checkpoint_path_qed))
        model_qed.load_state_dict(torch.load(checkpoint_path_qed))
        parameters_qed =  model_qed.parameters()
        print("Loading paramaters from {}".format(checkpoint_path_plogp))
        model_plogp.load_state_dict(torch.load(checkpoint_path_plogp))
        parameters_plogp =  model_plogp.parameters()
        
        ### Initialization
        print("Initializing samples...")
        gen_x = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)
        
        gen_x.requires_grad = True
        gen_adj.requires_grad = True
        requires_grad(parameters_qed, False)
        requires_grad(parameters_plogp, False)
        model_qed.eval()
        model_plogp.eval()
        
        noise_x = torch.randn_like(gen_x, device=self.device)
        noise_adj = torch.randn_like(gen_adj, device=self.device)
        
        ### Langevin dynamics
        print("Generating samples...")
        for _ in range(ld_step):
            noise_x.normal_(0, ld_noise)
            noise_adj.normal_(0, ld_noise)
            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)


            gen_out_qed = model_qed(gen_adj, gen_x)
            gen_out_plogp = model_plogp(gen_adj, gen_x)
            gen_out = 0.5 * gen_out_qed + 0.5 * gen_out_plogp
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