import os
import torch
import torch.nn as nn
from rdkit import Chem
from dig.ggraph.method import Generator
from .model import GraphFlowModel, GraphFlowModel_rl, GraphFlowModel_con_rl
from .train_utils import adjust_learning_rate, DataIterator


class GraphAF(Generator):
    r"""
        The method class for GraphAF algorithm proposed in the paper `GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation <https://arxiv.org/abs/2001.09382>`_. This class provides interfaces for running random generation, property
        optimization, and constrained optimization with GraphAF. Please refer to the `benchmark codes <https://github.com/divelab/DIG/tree/dig/benchmarks/ggraph/GraphAF>`_ for usage examples.
    """
    def __init__(self):
        super(GraphAF, self).__init__()
        self.model = None
    

    def get_model(self, task, model_conf_dict, checkpoint_path=None):
        if model_conf_dict['use_gpu'] and not torch.cuda.is_available():
            model_conf_dict['use_gpu'] = False
        if task == 'rand_gen':
            self.model = GraphFlowModel(model_conf_dict)
        elif task == 'prop_opt':
            self.model = GraphFlowModel_rl(model_conf_dict)
        elif task == 'const_opt':
            self.model = GraphFlowModel_con_rl(model_conf_dict)
        else:
            raise ValueError('Task {} is not supported in GraphDF!'.format(task))
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
    

    def load_pretrain_model(self, path):
        load_key = torch.load(path)
        for key in load_key.keys():
            if key in self.model.state_dict().keys():
                self.model.state_dict()[key].copy_(load_key[key].detach().clone())


    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict, save_interval, save_dir):
        r"""
            Running training for random generation task.
            
            Args:
                loader: The data loader for loading training samples. It is supposed to use dig.ggraph.dataset.QM9/ZINC250k
                    as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                lr (float): The learning rate for training.
                wd (float): The weight decay factor for training.
                max_epochs (int): The maximum number of training epochs.
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                save_interval (int): Indicate the frequency to save the model parameters to .pth files,
                    *e.g.*, if save_interval=2, the model parameters will be saved for every 2 training epochs.
                save_dir (str): The directory to save the model parameters.
        """

        self.get_model('rand_gen', model_conf_dict)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for epoch in range(1, max_epochs+1):
            total_loss = 0
            for batch, data_batch in enumerate(loader):
                optimizer.zero_grad()
                inp_node_features = data_batch.x #(B, N, node_dim)
                inp_adj_features = data_batch.adj #(B, 4, N, N)
                if model_conf_dict['use_gpu']:
                    inp_node_features = inp_node_features.cuda()
                    inp_adj_features = inp_adj_features.cuda()
                
                out_z, out_logdet = self.model(inp_node_features, inp_adj_features)
                loss = self.model.log_prob(out_z, out_logdet)
                loss.backward()
                optimizer.step()

                total_loss += loss.to('cpu').item()
                print('Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

            avg_loss = total_loss / (batch + 1)
            print("Training | Average loss {}".format(avg_loss))
            
            if epoch % save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'rand_gen_ckpt_{}.pth'.format(epoch)))


    def run_rand_gen(self, model_conf_dict, checkpoint_path, n_mols=100, num_min_node=7, num_max_node=25, temperature=0.75, atomic_num_list=[6, 7, 8, 9]):
        r"""
            Running graph generation for random generation task.
            
            Args:
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                checkpoint_path (str): The path to the saved model checkpoint file.
                n_mols (int, optional): The number of molecules to generate. (default: :obj:`100`)
                num_min_node (int, optional): The minimum number of nodes in the generated molecular graphs. (default: :obj:`7`)
                num_max_node (int, optional): The maximum number of nodes in the generated molecular graphs. (default: :obj:`25`)
                temperature (float, optional): A float numbers, the temperature parameter of prior distribution. (default: :obj:`0.75`)
                atomic_num_list (list, optional): A list of integers, the list of atomic numbers indicating the node types in the generated molecular graphs. (default: :obj:`[6, 7, 8, 9]`)
            
            :rtype:
                (all_mols, pure_valids),
                all_mols is a list of generated molecules represented by rdkit Chem.Mol objects;
                pure_valids is a list of integers, all are 0 or 1, indicating whether bond resampling happens.
        """
        
        self.get_model('rand_gen', model_conf_dict, checkpoint_path)
        self.model.eval()
        all_mols, pure_valids = [], []
        cnt_mol = 0

        while cnt_mol < n_mols:
            mol, no_resample, num_atoms = self.model.generate(atom_list=atomic_num_list, min_atoms=num_min_node, max_atoms=num_max_node, temperature=temperature)
            if (num_atoms >= num_min_node):
                cnt_mol += 1
                all_mols.append(mol)
                pure_valids.append(no_resample)
                if cnt_mol % 10 == 0:
                    print('Generated {} molecules'.format(cnt_mol))
        
        assert cnt_mol == n_mols, 'number of generated molecules does not equal num'        
        return all_mols, pure_valids


    def train_prop_optim(self, lr, wd, max_iters, warm_up, model_conf_dict, pretrain_path, save_interval, save_dir):
        r"""
            Running fine-tuning for property optimization task.
            
            Args:
                lr (float): The learning rate for fine-tuning.
                wd (float): The weight decay factor for training.
                max_iters (int): The maximum number of training iters.
                warm_up (int): The number of linear warm-up iters.
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                pretrain_path (str): The path to the saved pretrained model file.
                save_interval (int): Indicate the frequency to save the model parameters to .pth files,
                    *e.g.*, if save_interval=20, the model parameters will be saved for every 20 training iters.
                save_dir (str): The directory to save the model parameters.
        """
        
        
        self.get_model('prop_opt', model_conf_dict)
        self.load_pretrain_model(pretrain_path)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        print('start finetuning model(reinforce)')
        moving_baseline = None
        for cur_iter in range(max_iters):
            optimizer.zero_grad()    
            loss, per_mol_reward, per_mol_property_score, moving_baseline = self.model.reinforce_forward_optim(in_baseline=moving_baseline, cur_iter=cur_iter)

            num_mol = len(per_mol_reward)
            avg_reward = sum(per_mol_reward) / num_mol
            avg_score = sum(per_mol_property_score) / num_mol     
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), 1.0)
            adjust_learning_rate(optimizer, cur_iter, lr, warm_up)
            optimizer.step()

            print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, avg_reward, avg_score, loss.item()))

            if cur_iter % save_interval == save_interval - 1:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'prop_opt_net_{}.pth'.format(cur_iter)))

        print("Finetuning (Reinforce) Finished!")
    

    def run_prop_optim(self, model_conf_dict, checkpoint_path, n_mols=100, num_min_node=7, num_max_node=25, temperature=0.75, atomic_num_list=[6, 7, 8, 9]):
        r"""
            Running graph generation for property optimization task.
            
            Args:
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                checkpoint_path (str): The path to the saved model checkpoint file.
                n_mols (int, optional): The number of molecules to generate. (default: :obj:`100`)
                num_min_node (int, optional): The minimum number of nodes in the generated molecular graphs. (default: :obj:`7`)
                num_max_node (int, optional): The maximum number of nodes in the generated molecular graphs. (default: :obj:`25`)
                temperature (float, optional): A float numbers, the temperature parameter of prior distribution. (default: :obj:`0.75`)
                atomic_num_list (list, optional): A list of integers, the list of atomic numbers indicating the node types in the generated molecular graphs. (default: :obj:`[6, 7, 8, 9]`)
            
            :rtype:
                all_mols, a list of generated molecules represented by rdkit Chem.Mol objects.
        """
        
        self.get_model('prop_opt', model_conf_dict, checkpoint_path)
        self.model.eval()
        all_mols, all_smiles = [], []
        cnt_mol = 0

        while cnt_mol < n_mols:
            mol, num_atoms = self.model.reinforce_optim_one_mol(atom_list=atomic_num_list, max_size_rl=num_max_node, temperature=temperature)
            if mol is not None:
                smile = Chem.MolToSmiles(mol)
                if num_atoms >= num_min_node and not smile in all_smiles:
                    all_mols.append(mol)
                    all_smiles.append(smile)
                    cnt_mol += 1
                    if cnt_mol % 10 == 0:
                        print('Generated {} molecules'.format(cnt_mol))
        
        return all_mols


    def train_cons_optim(self, loader, lr, wd, max_iters, warm_up, model_conf_dict, pretrain_path, save_interval, save_dir):
        r"""
            Running fine-tuning for constrained optimization task.
            
            Args:
                loader: The data loader for loading training samples. It is supposed to use dig.ggraph.dataset.ZINC800
                    as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                lr (float): The learning rate for training.
                wd (float): The weight decay factor for training.
                max_iters (int): The maximum number of training iters.
                warm_up (int): The number of linear warm-up iters.
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                pretrain_path (str): The path to the saved pretrained model parameters file.
                save_interval (int): Indicate the frequency to save the model parameters to .pth files,
                    *e.g.*, if save_interval=20, the model parameters will be saved for every 20 training iters.
                save_dir (str): The directory to save the model parameters.
        """
        
        self.get_model('const_prop_opt', model_conf_dict)
        self.load_pretrain_model(pretrain_path)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        loader = DataIterator(loader)

        print('start finetuning model(reinforce)')
        moving_baseline = None
        for cur_iter in range(max_iters):
            optimizer.zero_grad()
            batch_data = next(loader)
            mol_xs = batch_data.x
            mol_adjs = batch_data.adj
            mol_sizes = batch_data.num_atom
            bfs_perm_origin = batch_data.bfs_perm_origin
            raw_smiles = batch_data.smile
        
            loss, per_mol_reward, per_mol_property_score, moving_baseline = self.model.reinforce_forward_constrained_optim(
                                                    mol_xs=mol_xs, mol_adjs=mol_adjs, mol_sizes=mol_sizes, raw_smiles=raw_smiles, 
                                                    bfs_perm_origin=bfs_perm_origin, in_baseline=moving_baseline, cur_iter=cur_iter)

            num_mol = len(per_mol_reward)
            avg_reward = sum(per_mol_reward) / num_mol
            avg_score = sum(per_mol_property_score) / num_mol
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), 1.0)
            adjust_learning_rate(optimizer, cur_iter, lr, warm_up)
            optimizer.step()

            print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, avg_reward, avg_score, loss.item()))

            if cur_iter % save_interval == save_interval - 1:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'const_prop_opt_net_{}.pth'.format(cur_iter)))

        print("Finetuning (Reinforce) Finished!")
    

    def run_cons_optim_one_mol(self, adj, x, org_smile, mol_size, bfs_perm_origin, max_size_rl=38, temperature=0.70, atom_list=[6, 7, 8, 9]):
        
        best_mol0 = None
        best_mol2 = None
        best_mol4 = None
        best_mol6 = None
        best_imp0 = -100.
        best_imp2 = -100.
        best_imp4 = -100.
        best_imp6 = -100.
        final_sim0 = -1.
        final_sim2 = -1.
        final_sim4 = -1.
        final_sim6 = -1.

        mol_org = Chem.MolFromSmiles(org_smile)
        mol_org_size = mol_org.GetNumAtoms()
        assert mol_org_size == mol_size

        cur_mols, cur_mol_imps, cur_mol_sims = self.model.reinforce_constrained_optim_one_mol(x, adj, mol_size, org_smile, bfs_perm_origin,
                                                                        atom_list=atom_list, temperature=temperature, max_size_rl=max_size_rl)
        num_success = len(cur_mol_imps)
        for i in range(num_success):
            cur_mol = cur_mols[i]
            cur_imp = cur_mol_imps[i]
            cur_sim = cur_mol_sims[i]
            assert cur_imp > 0
            if cur_sim > 0:
                if cur_imp > best_imp0:
                    best_mol0 = cur_mol
                    best_imp0 = cur_imp
                    final_sim0 = cur_sim
            if cur_sim > 0.2:
                if cur_imp > best_imp2:
                    best_mol2 = cur_mol
                    best_imp2 = cur_imp
                    final_sim2 = cur_sim
            if cur_sim > 0.4:
                if cur_imp > best_imp4:
                    best_mol4 = cur_mol
                    best_imp4 = cur_imp
                    final_sim4 = cur_sim
            if cur_sim > 0.6:
                if cur_imp > best_imp6:
                    best_mol6 = cur_mol
                    best_imp6 = cur_imp
                    final_sim6 = cur_sim                    

        return [best_mol0, best_mol2, best_mol4, best_mol6], [best_imp0, best_imp2, best_imp4, best_imp6], [final_sim0, final_sim2, final_sim4, final_sim6]


    def run_cons_optim(self, dataset, model_conf_dict, checkpoint_path, repeat_time=200, min_optim_time=50, num_max_node=25, temperature=0.7, atomic_num_list=[6, 7, 8, 9]):
        r"""
            Running molecule optimization for constrained optimization task.
            
            Args:
                dataset: The dataset class for loading molecules to be optimized. It is supposed to use dig.ggraph.dataset.ZINC800 as the dataset class.
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                checkpoint_path (str): The path to the saved model checkpoint file.
                repeat_time (int, optional): The maximum number of optimization times for each molecule before successfully optimizing it under the threshold 0.6.  (default: :obj:`200`)
                min_optim_time (int, optional): The minimum number of optimization times for each molecule. (default: :obj:`50`)
                num_max_node (int, optional): The maximum number of nodes in the optimized molecular graphs. (default: :obj:`25`)
                temperature (float, optional): A float numbers, the temperature parameter of prior distribution. (default: :obj:`0.75`)
                atomic_num_list (list, optional): A list of integers, the list of atomic numbers indicating the node types in the optimized molecular graphs. (default: :obj:`[6, 7, 8, 9]`)
            
            :rtype:
                (mols_0, mols_2, mols_4, mols_6), they are lists of optimized molecules (represented by rdkit Chem.Mol objects) under the threshold 0.0, 0.2, 0.4, 0.6, respectively.
        """
        
        
        self.get_model('const_prop_opt', model_conf_dict, checkpoint_path)
        self.model.eval()

        data_len = len(dataset)
        optim_success_dict = {}
        mols_0, mols_2, mols_4, mols_6 = [], [], [], []
        for batch_cnt in range(data_len):
            best_mol = [None, None, None, None]
            best_score = [-100., -100., -100., -100.]
            final_sim = [-1., -1., -1., -1.]

            batch_data = dataset[batch_cnt] # dataloader is dataset object

            inp_node_features = batch_data.x.unsqueeze(0) #(1, N, node_dim)              
            inp_adj_features = batch_data.adj.unsqueeze(0) #(1, 4, N, N)              

            raw_smile = batch_data.smile  #(1)
            mol_size = batch_data.num_atom
            bfs_perm_origin = batch_data.bfs_perm_origin

            for cur_iter in range(repeat_time):
                if raw_smile not in optim_success_dict:
                    optim_success_dict[raw_smile] = [0, -1] #(try_time, imp)
                if optim_success_dict[raw_smile][0] > min_optim_time and optim_success_dict[raw_smile][1] > 0: # reach min time and imp is positive
                    continue # not optimize this one

                best_mol0246, best_score0246, final_sim0246 = self.run_cons_optim_one_mol(inp_adj_features, 
                                                                    inp_node_features, raw_smile, mol_size, bfs_perm_origin, num_max_node, temperature, atomic_num_list)
                if best_score0246[0] > best_score[0]:
                    best_score[0] = best_score0246[0]
                    best_mol[0] = best_mol0246[0]
                    final_sim[0] = final_sim0246[0]

                if best_score0246[1] > best_score[1]:
                    best_score[1] = best_score0246[1]
                    best_mol[1] = best_mol0246[1]
                    final_sim[1] = final_sim0246[1] 

                if best_score0246[2] > best_score[2]:
                    best_score[2] = best_score0246[2]
                    best_mol[2] = best_mol0246[2]
                    final_sim[2] = final_sim0246[2]
                    
                if best_score0246[3] > best_score[3]:
                    best_score[3] = best_score0246[3]
                    best_mol[3] = best_mol0246[3]
                    final_sim[3] = final_sim0246[3]

                if best_score[3] > 0: #imp > 0
                    optim_success_dict[raw_smile][1] = best_score[3]
                optim_success_dict[raw_smile][0] += 1 # try time + 1

            mols_0.append(best_mol[0])
            mols_2.append(best_mol[1])
            mols_4.append(best_mol[2])
            mols_6.append(best_mol[3])

            if batch_cnt % 1 == 0:
                print('Optimized {} molecules'.format(batch_cnt+1))

        return mols_0, mols_2, mols_4, mols_6
