import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from dig.ggraph.utils import check_chemical_validity, check_valency, calculate_min_plogp, reward_target_molecule_similarity
from dig.ggraph.utils import convert_radical_electrons_to_hydrogens, steric_strain_filter, zinc_molecule_filter
from .disgraphaf import DisGraphAF


class GraphFlowModel_con_rl(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel_con_rl, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.node_dim = model_conf_dict['node_dim']
        self.bond_dim = model_conf_dict['bond_dim']
        self.edge_unroll = model_conf_dict['edge_unroll']
        self.conf_rl = model_conf_dict['rl_conf_dict']

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(0)  # (max_size) + (max_edge_unroll - 1) / 2 * max_edge_unroll + (max_size - max_edge_unroll) * max_edge_unroll
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim

        self.dp = model_conf_dict['use_gpu']
        
        node_base_log_probs = torch.randn(self.max_size, self.node_dim)
        edge_base_log_probs = torch.randn(self.latent_step - self.max_size, self.bond_dim)
        self.flow_core = DisGraphAF(node_masks, adj_masks, link_prediction_index, num_flow_layer = model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                    num_node_type=self.node_dim, num_edge_type=self.bond_dim, num_rgcn_layer=model_conf_dict['num_rgcn_layer'], nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])
        self.flow_core_old = DisGraphAF(node_masks, adj_masks, link_prediction_index, num_flow_layer = model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                    num_node_type=self.node_dim, num_edge_type=self.bond_dim, num_rgcn_layer=model_conf_dict['num_rgcn_layer'], nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)
            self.flow_core_old = nn.DataParallel(self.flow_core_old)
            self.node_base_log_probs = nn.Parameter(node_base_log_probs.cuda(), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.cuda(), requires_grad=True)
            self.node_base_log_probs_old = nn.Parameter(node_base_log_probs.cuda(), requires_grad=False)
            self.edge_base_log_probs_old = nn.Parameter(edge_base_log_probs.cuda(), requires_grad=False)
        else:
            self.node_base_log_probs = nn.Parameter(node_base_log_probs, requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs, requires_grad=True)
            self.node_base_log_probs_old = nn.Parameter(node_base_log_probs, requires_grad=False)
            self.edge_base_log_probs_old = nn.Parameter(edge_base_log_probs, requires_grad=False)
        

    def reinforce_constrained_optim_one_mol(self, x, adj, mol_size, raw_smile, bfs_perm_origin, atom_list, temperature=[0.3,0.3], max_size_rl=38):
        num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2atom = {i:atom_list[i] for i in range(len(atom_list))}

        self.eval()
        
        cur_mols = []
        cur_mol_imps = []
        cur_mol_sims = []

        with torch.no_grad():
            flag_reconstruct_from_node_adj = True

            rand_num = np.random.rand()
            if rand_num <= 0.5:
                cur_modify_size = np.random.randint(low=0, high=self.conf_rl['modify_size'])
            else:
                cur_modify_size = 0

            keep_size = mol_size - cur_modify_size
            
            org_bfs_perm_origin = bfs_perm_origin
            org_node_features = x #(1, 38, 9)
            org_adj_features = adj # (1, 4, 38, 38)

            cur_node_features = x.clone() #(1, 38, 9)
            cur_adj_features = adj.clone() # (1, 4, 38, 38)                

            cur_node_features[:, keep_size:, :] = 0.
            cur_adj_features[:, :, keep_size:, :] = 0.
            cur_adj_features[:, :, :, keep_size:] = 0.

            rw_mol = Chem.RWMol()  # editable mol
            mol = None
            
            for i in range(mol_size):
                node_id = torch.argmax(org_node_features[0, i]).item()  #(9, )
                if i < keep_size:
                    rw_mol.AddAtom(Chem.Atom(num2atom[node_id]))
                else:
                    pass

                if i < self.edge_unroll:
                    edge_total = i  # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll

                for j in range(edge_total):
                    edge_id = torch.argmax(org_adj_features[0, :, i, j+start]).item() #(4, )
                    if edge_id == 3: # no bond to add
                        continue
                    if i < keep_size:
                        rw_mol.AddBond(i, j + start, num2bond[edge_id])
                    else:
                        pass

            mol = rw_mol.GetMol()
            s_raw = raw_smile
            org_mol_true_raw = Chem.MolFromSmiles(s_raw)

            #calculate property for mol org
            org_mol_plogp = calculate_min_plogp(org_mol_true_raw)

            if check_chemical_validity(mol) is False or (cur_modify_size == 0 and np.random.rand() <= 0.5): # do not use subgraph, use original graph
                rw_mol = Chem.RWMol(org_mol_true_raw)
                mol = rw_mol.GetMol()
                flag_reconstruct_from_node_adj = False # bfs is not reflected in current mol
                keep_size = mol_size
                cur_node_features = org_node_features.clone()
                cur_adj_features = org_adj_features.clone()
                cur_node_features[:, keep_size:, :] = 0.
                cur_adj_features[:, :, keep_size:, :] = 0.
                cur_adj_features[:, :, :, keep_size:] = 0.

            assert check_chemical_validity(org_mol_true_raw) is True, 's_raw is %s' % (s_raw)
            assert check_chemical_validity(mol) is True

            assert mol.GetNumAtoms() == keep_size
            assert org_mol_true_raw.GetNumAtoms() == mol_size

            if self.dp:
                cur_node_features = cur_node_features.cuda()
                cur_adj_features = cur_adj_features.cuda()

            is_continue = True
            if keep_size <= self.edge_unroll:
                edge_idx = int(keep_size * (keep_size - 1) / 2)
            else:
                edge_idx = int(self.edge_unroll * (self.edge_unroll - 1) / 2 + (keep_size - self.edge_unroll) * self.edge_unroll)

            min_action_node = max(1, max_size_rl - keep_size)
            added_num = 0
            node_features_each_iter_backup = cur_node_features.clone() # backup of features, updated when newly added node is connected to previous subgraph
            adj_features_each_iter_backup = cur_adj_features.clone()
            for i in range(keep_size, max_size_rl):
                if not is_continue:
                    break

                if i < self.edge_unroll:
                    edge_total = i  # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll
                
                # first generate node
                ## reverse flow
                prior_node_dist = torch.distributions.OneHotCategorical(logits=self.node_base_log_probs[i]*temperature[0])
                latent_node = prior_node_dist.sample().view(1, -1)

                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_node, mode=0).view(-1)  # (9, )
                else:
                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_node, mode=0).view(-1)  # (9, )
               
                feature_id = torch.argmax(latent_node).item()
               
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                # cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    valid = False
                    resample_edge = 0
                    edge_dis = self.edge_base_log_probs[edge_idx].clone()
                    invalid_bond_type_set = set()
                    while not valid:
                        if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                            prior_edge_dist = torch.distributions.OneHotCategorical(logits=edge_dis/temperature[1])
                            latent_edge = prior_edge_dist.sample().view(1, -1)
                            latent_id = torch.argmax(latent_edge, dim=1)

                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge,
                                                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                            else:
                                latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_edge, mode=1,
                                                            edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:
                            assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                            edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                        
                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                        if edge_discrete_id == 3:  # virtual edge
                            valid = True # virtual edge is alway valid
                        else:  # single/double/triple bond
                            if flag_reconstruct_from_node_adj:
                                rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                            else:
                                #current rw_mol is copy from original_mol, which does not reflect the bfs order
                                #so we use bfs_perm_origin to map j+start to the true node_id in rw_mol
                                rw_mol.AddBond(i, int(org_bfs_perm_origin[j+start].item()), num2bond[edge_discrete_id])
                            valid = check_valency(rw_mol)
                            if valid:
                                is_connect = True
                            else:  # backtrack
                                edge_dis[latent_id] = float('-inf')
                                if flag_reconstruct_from_node_adj:
                                    rw_mol.RemoveBond(i, j + start)
                                else:
                                    rw_mol.RemoveBond(i, int(org_bfs_perm_origin[j+start].item()))
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                resample_edge += 1

                                invalid_bond_type_set.add(edge_discrete_id)

                    edge_idx += 1    
                            
                if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    mol = rw_mol.GetMol()
                    
                    if check_chemical_validity(mol) is True:
                        current_smile = Chem.MolToSmiles(mol, isomericSmiles=True)
                        tmp_mol1 = Chem.MolFromSmiles(current_smile)
                        current_imp = calculate_min_plogp(tmp_mol1) - org_mol_plogp
                        current_sim = reward_target_molecule_similarity(tmp_mol1, org_mol_true_raw)
                        if current_imp > 0:
                            cur_mols.append(tmp_mol1)
                            cur_mol_imps.append(current_imp)
                            cur_mol_sims.append(current_sim)

                        if flag_reconstruct_from_node_adj is False: #not reconstructed from adj, can convert.
                            mol_converted = convert_radical_electrons_to_hydrogens(mol)                                        
                            if check_chemical_validity(mol_converted) is True:
                                current_smile2 = Chem.MolToSmiles(mol_converted, isomericSmiles=True)
                                tmp_mol2 = Chem.MolFromSmiles(current_smile2)
                                current_imp2 = calculate_min_plogp(tmp_mol2) - org_mol_plogp
                                current_sim2 = reward_target_molecule_similarity(tmp_mol2, org_mol_true_raw)
                                if current_imp2 > 0:
                                    cur_mols.append(tmp_mol2)
                                    cur_mol_imps.append(current_imp2)
                                    cur_mol_sims.append(current_sim2)

                    node_features_each_iter_backup = cur_node_features.clone() # update node backup since new node is valid
                    adj_features_each_iter_backup = cur_adj_features.clone()               
                    added_num += 1

                else:
                    #! we may need to satisfy min action here
                    if added_num >= min_action_node:
                        # if we have already add 'min_action_node', ignore and let continue be false.
                        # the data added in last iter will be pop afterwards
                        is_continue = False
                    else:
                        is_continue = False # first set as False

                        rw_mol = Chem.RWMol(mol) # important, recover the mol
                        cur_node_features = node_features_each_iter_backup.clone() # recover the backuped features
                        cur_adj_features = adj_features_each_iter_backup.clone()
                        cur_node_features_tmp = cur_node_features.clone()
                        cur_adj_features_tmp = cur_adj_features.clone()
                        cur_mol_size = rw_mol.GetNumAtoms()

                        mol_demon_edit = Chem.RWMol(rw_mol)
                        last_id2 = mol_demon_edit.AddAtom(Chem.Atom(6))
                        cur_node_features_tmp[0, last_id2, 0] = 1.0
                        cur_adj_features_tmp[0, :, last_id2, last_id2] = 1.0
                        
                        flag_success = False
                        if cur_mol_size > keep_size:
                            mol_demon_edit.AddBond(cur_mol_size-1, last_id2, Chem.rdchem.BondType.SINGLE)
                            cur_adj_features_tmp[0, 0, cur_mol_size-1, last_id2] = 1.0
                            cur_adj_features_tmp[0, 0, last_id2, cur_mol_size-1] = 1.0
                            valid = check_valency(mol_demon_edit)
                            if valid:
                                flag_success = True
                            else:
                                mol_demon_edit.RemoveBond(cur_mol_size-1, last_id2)
                                cur_adj_features_tmp[0, 0, cur_mol_size-1, last_id2] = 0.0
                                cur_adj_features_tmp[0, 0, last_id2, cur_mol_size-1] = 0.0

                                count = 0
                                while True:
                                    if count > 100:
                                        break
                                    if last_id2 > 12:
                                        j = np.random.randint(1, 13)
                                    else:
                                        j = np.random.randint(1, last_id2 + 1)
                                    if flag_reconstruct_from_node_adj:
                                        mol_demon_edit.AddBond(int(last_id2 - j), int(last_id2), Chem.rdchem.BondType.SINGLE)
                                    else:
                                        mol_demon_edit.AddBond(int(org_bfs_perm_origin[last_id2 - j]), int(last_id2), Chem.rdchem.BondType.SINGLE)
                                    cur_adj_features_tmp[0, 0, last_id2 - j, last_id2] = 1.0
                                    cur_adj_features_tmp[0, 0, last_id2, last_id2 - j] = 1.0

                                    valid = check_valency(mol_demon_edit)
                                    if valid:
                                        flag_success = True
                                        break
                                    else:
                                        if flag_reconstruct_from_node_adj:
                                            mol_demon_edit.RemoveBond(int(last_id2 - j), int(last_id2))
                                        else:
                                            mol_demon_edit.RemoveBond(int(org_bfs_perm_origin[last_id2 - j]), int(last_id2))
                                        cur_adj_features_tmp[0, 0, last_id2 - j, last_id2] = 0.0
                                        cur_adj_features_tmp[0, 0, last_id2, last_id2 - j] = 0.0
                                        count += 1
                        else:
                            count = 0
                            while True:
                                if count > 100:
                                    break
                                if keep_size > 12:
                                    j = np.random.randint(1, 13)
                                else:
                                    j = np.random.randint(1, keep_size+1)
                                if flag_reconstruct_from_node_adj:
                                    mol_demon_edit.AddBond(int(keep_size-j), int(keep_size), Chem.rdchem.BondType.SINGLE)
                                else:
                                    mol_demon_edit.AddBond(int(org_bfs_perm_origin[keep_size-j]), int(keep_size), Chem.rdchem.BondType.SINGLE)
                                cur_adj_features_tmp[0, 0, keep_size - j, keep_size] = 1.0
                                cur_adj_features_tmp[0, 0, keep_size, keep_size - j] = 1.0

                                valid = check_valency(mol_demon_edit)
                                if valid:
                                    flag_success = True
                                    break
                                else:
                                    if flag_reconstruct_from_node_adj:
                                        mol_demon_edit.RemoveBond(int(keep_size-j), int(keep_size))
                                    else:
                                        mol_demon_edit.RemoveBond(int(org_bfs_perm_origin[keep_size-j]), int(keep_size))
                                    cur_adj_features_tmp[0, 0, keep_size - j, keep_size] = 0.0
                                    cur_adj_features_tmp[0, 0, keep_size, keep_size - j] = 0.0
                                    count += 1


                        if flag_success and check_chemical_validity(mol_demon_edit): # successfully take one min action.
                            rw_mol = Chem.RWMol(mol_demon_edit)
                            cur_node_features = cur_node_features_tmp.clone()
                            cur_adj_features = cur_adj_features_tmp.clone()
                            is_continue = True
                            mol = rw_mol.GetMol()
                            if check_chemical_validity(mol) is True:
                                current_smile = Chem.MolToSmiles(mol, isomericSmiles=True)
                                tmp_mol1 = Chem.MolFromSmiles(current_smile)
                                current_imp = calculate_min_plogp(tmp_mol1) - org_mol_plogp
                                current_sim = reward_target_molecule_similarity(tmp_mol1, org_mol_true_raw)
                                if current_imp > 0:
                                    cur_mols.append(tmp_mol1)
                                    cur_mol_imps.append(current_imp)
                                    cur_mol_sims.append(current_sim)

                                if flag_reconstruct_from_node_adj is False: #not reconstructed from adj, can convert.
                                    mol_converted = convert_radical_electrons_to_hydrogens(mol)                                        
                                    if check_chemical_validity(mol_converted) is True:
                                        current_smile2 = Chem.MolToSmiles(mol_converted, isomericSmiles=True)
                                        tmp_mol2 = Chem.MolFromSmiles(current_smile2)
                                        current_imp2 = calculate_min_plogp(tmp_mol2) - org_mol_plogp
                                        current_sim2 = reward_target_molecule_similarity(tmp_mol2, org_mol_true_raw)
                                        if current_imp2 > 0:
                                            cur_mols.append(tmp_mol2)
                                            cur_mol_imps.append(current_imp2)
                                            cur_mol_sims.append(current_sim2)                                     
                            
                            node_features_each_iter_backup = cur_node_features.clone() # update node backup since new node is valid
                            adj_features_each_iter_backup = cur_adj_features.clone() #
                            added_num += 1                                        
                                                        
        return cur_mols, cur_mol_imps, cur_mol_sims


    def reinforce_forward_constrained_optim(self, mol_xs, mol_adjs, mol_sizes, raw_smiles, bfs_perm_origin, in_baseline=None, cur_iter=None):
        """
        Fintuning model using reinforce algorithm
        Args:
            existing_mol: molecule to be optimized. Practically, we provide 64 mols per call and the function may take less then 64 mols
            temperature: generation temperature
            batch_size: batch_size for collecting data
            max_size_rl: maximal num of atoms allowed for generation

        Returns:

        """
        assert cur_iter is not None
        atom_list, temperature, batch_size, max_size_rl = self.conf_rl['atom_list'], self.conf_rl['temperature'], self.conf_rl['batch_size'], self.conf_rl['max_size_rl']
        batch_size = min(batch_size, mol_sizes.size(0)) # last batch of one iter may be less batch_size
        if cur_iter % self.conf_rl['update_iters'] == 0: # uodate the demenstration net every 4 iter.
            print('copying to old model at iter {}'.format(cur_iter))
            self.flow_core_old.load_state_dict(self.flow_core.state_dict())
            self.node_base_log_probs_old = nn.Parameter(self.node_base_log_probs.detach().clone(), requires_grad=False)
            self.edge_base_log_probs_old = nn.Parameter(self.edge_base_log_probs.detach().clone(), requires_grad=False)

        #assert cur_baseline is not None
        num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2atom = {i:atom_list[i] for i in range(len(atom_list))}

        node_inputs = {}
        node_inputs['node_features'] = []
        node_inputs['adj_features'] = []
        node_inputs['node_features_cont'] = []
        node_inputs['rewards'] = []
        node_inputs['baseline_index'] = []
        node_inputs['node_cnt'] = []

        adj_inputs = {}
        adj_inputs['node_features'] = []
        adj_inputs['adj_features'] = []
        adj_inputs['edge_features_cont'] = []
        adj_inputs['index'] = []
        adj_inputs['rewards'] = []
        adj_inputs['baseline_index'] = []
        adj_inputs['edge_cnt'] = []

        reward_baseline = torch.zeros([max_size_rl + 5, 2]).cuda()

        max_action_size = batch_size * (int(max_size_rl + (self.edge_unroll - 1) * self.edge_unroll / 2 + (max_size_rl-self.edge_unroll) * self.edge_unroll))

        batch_length = 0
        total_node_step = 0
        total_edge_step = 0

        per_mol_reward = []
        per_mol_property_score = []
        
        ### gather training data from generation
        self.eval() #! very important. Because we use batch normalization, training mode will result in unrealistic molecules
        
        with torch.no_grad():
            while total_node_step + total_edge_step < max_action_size and batch_length < batch_size:                
                traj_node_inputs = {}
                traj_node_inputs['node_features'] = []
                traj_node_inputs['adj_features'] = []
                traj_node_inputs['node_features_cont'] = []
                traj_node_inputs['rewards'] = []
                traj_node_inputs['baseline_index'] = []
                traj_node_inputs['node_cnt'] = []
                traj_adj_inputs = {}
                traj_adj_inputs['node_features'] = []
                traj_adj_inputs['adj_features'] = []
                traj_adj_inputs['edge_features_cont'] = []
                traj_adj_inputs['index'] = []
                traj_adj_inputs['rewards'] = []
                traj_adj_inputs['baseline_index'] = []
                traj_adj_inputs['edge_cnt'] = []

                step_cnt = 1.0

                flag_reconstruct_from_node_adj = True

                rand_num = np.random.rand()
                if rand_num <= 0.5:
                    cur_modify_size = np.random.randint(low=0, high=self.conf_rl['modify_size'])
                else:
                    cur_modify_size = 0
                
                keep_size = int(mol_sizes[batch_length]) - cur_modify_size
                
                org_bfs_perm_origin = bfs_perm_origin[batch_length]
                org_node_features = mol_xs[batch_length:batch_length+1] #(1, 38, 9)
                org_adj_features = mol_adjs[batch_length:batch_length+1] # (1, 4, 38, 38)

                cur_node_features = mol_xs[batch_length:batch_length+1].clone() #(1, 38, 9)
                cur_adj_features = mol_adjs[batch_length:batch_length+1].clone() # (1, 4, 38, 38)                

                cur_node_features[:, keep_size:, :] = 0.
                cur_adj_features[:, :, keep_size:, :] = 0.
                cur_adj_features[:, :, :, keep_size:] = 0.

                min_action_node = 1

                rw_mol = Chem.RWMol()  # editable mol
                mol = None
                
                for i in range(mol_sizes[batch_length]):
                    node_id = torch.argmax(org_node_features[0, i]).item()  #(9, )
                    if i < keep_size:
                        rw_mol.AddAtom(Chem.Atom(num2atom[node_id]))
                    else:
                        pass

                    if i < self.edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll

                    for j in range(edge_total):
                        edge_id = torch.argmax(org_adj_features[0, :, i, j+start]).item() #(4, )
                        if edge_id == 3: # no bond to add
                            continue
                        if i < keep_size:
                            rw_mol.AddBond(i, j + start, num2bond[edge_id])
                        else:
                            pass

                mol = rw_mol.GetMol()
                s_raw = raw_smiles[batch_length]
                org_mol_true_raw = Chem.MolFromSmiles(s_raw)

                if check_chemical_validity(mol) is False: # do not use subgraph, use original graph
                    rw_mol = Chem.RWMol(org_mol_true_raw)
                    mol = rw_mol.GetMol()
                    flag_reconstruct_from_node_adj = False # bfs is not reflected in current mol
                    keep_size = int(mol_sizes[batch_length])
                    cur_node_features = org_node_features.clone()
                    cur_adj_features = org_adj_features.clone()
                    cur_node_features[:, keep_size:, :] = 0.
                    cur_adj_features[:, :, keep_size:, :] = 0.
                    cur_adj_features[:, :, :, keep_size:] = 0.                                        

                assert check_chemical_validity(org_mol_true_raw) is True, 's_raw is %s' % (s_raw)
                assert check_chemical_validity(mol) is True

                assert mol.GetNumAtoms() == keep_size
                assert org_mol_true_raw.GetNumAtoms() == mol_sizes[batch_length]

                if self.dp:
                    cur_node_features = cur_node_features.cuda()
                    cur_adj_features = cur_adj_features.cuda()

                is_continue = True
                if keep_size <= self.edge_unroll:
                    edge_idx = int(keep_size * (keep_size - 1) / 2)
                else:
                    edge_idx = int(self.edge_unroll * (self.edge_unroll - 1) / 2 + (keep_size - self.edge_unroll) * self.edge_unroll)

                step_num_data_edge = 0
                added_num = 0
                node_features_each_iter_backup = cur_node_features.clone()
                adj_features_each_iter_backup = cur_adj_features.clone()
                for i in range(keep_size, max_size_rl):
                    if not is_continue:
                        break                    
                    
                    step_num_data_edge = 0 # generating new node and its edges. Not sure if this will add into the final mol.

                    if i < self.edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll
                    
                    # first generate node
                    ## reverse flow
                    prior_node_dist = torch.distributions.OneHotCategorical(logits=self.node_base_log_probs_old[i]*temperature[0])
                    latent_node = prior_node_dist.sample().view(1, -1)
                    
                    if self.dp:
                        latent_node = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features, latent_node, mode=0).view(-1)  # (9, )
                    else:
                        latent_node = self.flow_core_old.reverse(cur_node_features, cur_adj_features, latent_node, mode=0).view(-1)  # (9, )
                    
                    feature_id = torch.argmax(latent_node).item()
                    total_node_step += 1
                    node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                    node_feature_cont[0, feature_id] = 1.0

                    # update traj inputs for node_id
                    traj_node_inputs['node_features'].append(cur_node_features.clone())  # (1, max_size_rl, self.node_dim)
                    traj_node_inputs['adj_features'].append(cur_adj_features.clone())  # (1, self.bond_dim, max_size_rl, max_size_rl)
                    traj_node_inputs['node_features_cont'].append(node_feature_cont)  # (1, self.node_dim)
                    traj_node_inputs['rewards'].append(torch.full(size=(1,1), fill_value=step_cnt).cuda())  # (1, 1)
                    traj_node_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1, 1)
                    traj_node_inputs['node_cnt'].append(torch.full(size=(1,), fill_value=float(i)).long().cuda())

                    cur_node_features[0, i, feature_id] = 1.0
                    cur_adj_features[0, :, i, i] = 1.0
                    rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                    # then generate edges
                    if i == 0:
                        is_connect = True
                    else:
                        is_connect = False
                    # cur_mol_size = mol.GetNumAtoms
                    for j in range(edge_total):
                        valid = False
                        resample_edge = 0
                        edge_dis = self.edge_base_log_probs_old[edge_idx].clone()
                        invalid_bond_type_set = set()
                        while not valid:
                            if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                                prior_edge_dist = torch.distributions.OneHotCategorical(logits=edge_dis/temperature[1])
                                latent_edge = prior_edge_dist.sample().view(1, -1)
                                latent_id = torch.argmax(latent_edge, dim=1)
                                
                                if self.dp:
                                    latent_edge = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features, latent_edge,
                                                                                mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                else:
                                    latent_edge = self.flow_core_old.reverse(cur_node_features, cur_adj_features, latent_edge, mode=1,
                                                                edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                edge_discrete_id = torch.argmax(latent_edge).item()
                            else:
                                assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                                edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                            
                            total_edge_step += 1
                            edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                            edge_feature_cont[0, edge_discrete_id] = 1.0

                            # update traj inputs for edge_id
                            traj_adj_inputs['node_features'].append(cur_node_features.clone())  # 1, max_size_rl, self.node_dim
                            traj_adj_inputs['adj_features'].append(cur_adj_features.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                            traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)  # 1, self.bond_dim
                            traj_adj_inputs['index'].append(torch.Tensor([[j + start, i]]).long().cuda().view(1,-1)) # (1, 2)
                            traj_adj_inputs['edge_cnt'].append(torch.full(size=(1,), fill_value=float(edge_idx)).long().cuda())
                            step_num_data_edge += 1 # add one edge data, not sure if this should be added to the final train data

                            cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                            cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                            if edge_discrete_id == 3:  # virtual edge
                                valid = True # virtual edge is alway valid
                            else:  # single/double/triple bond
                                if flag_reconstruct_from_node_adj:
                                    rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                                else:
                                    #current rw_mol is copy from original_mol, which does not reflect the bfs order
                                    #so we use bfs_perm_origin to map j+start to the true node_id in rw_mol
                                    rw_mol.AddBond(i, int(org_bfs_perm_origin[j+start].item()), num2bond[edge_discrete_id])
                                valid = check_valency(rw_mol)
                                if valid:
                                    is_connect = True
                                else:  # backtrack
                                    edge_dis[latent_id] = float('-inf')
                                    if flag_reconstruct_from_node_adj:
                                        rw_mol.RemoveBond(i, j + start)
                                    else:
                                        rw_mol.RemoveBond(i, int(org_bfs_perm_origin[j+start].item()))
                                    cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                    cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                    resample_edge += 1

                                    invalid_bond_type_set.add(edge_discrete_id)

                            if valid:
                                traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)
                            else:
                                if self.conf_rl['penalty']:
                                    traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=-1.).cuda())  # (1, 1) invalid edge penalty
                                    traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1,)
                                else:
                                    traj_adj_inputs['node_features'].pop(-1)
                                    traj_adj_inputs['adj_features'].pop(-1)
                                    traj_adj_inputs['edge_features_cont'].pop(-1)
                                    traj_adj_inputs['index'].pop(-1)
                                    traj_adj_inputs['edge_cnt'].pop(-1)
                                    step_num_data_edge -= 1 # if we do not penalize invalid edge, pop train data, decrease counter by 1                              

                        edge_idx += 1        

                    if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                        is_continue = True
                        mol = rw_mol.GetMol()
                        node_features_each_iter_backup = cur_node_features.clone()
                        adj_features_each_iter_backup = cur_adj_features.clone()
                        added_num += 1
                        #current_i_continue = False

                    else:
                        #! we may need to satisfy min action here
                        if added_num >= min_action_node:
                            # if we have already add 'min_action_node', ignore and let continue be false.
                            # the data added in last iter will be pop afterwards
                            is_continue = False
                        else:
                            # added_num is less than min_action_node
                            # first pop the appended train data
                            is_continue = False # first set as False
                            rw_mol = Chem.RWMol(mol) # important, recover the mol
                            cur_node_features = node_features_each_iter_backup.clone() # recover the backuped features
                            cur_adj_features = adj_features_each_iter_backup.clone()
                            cur_node_features_tmp = cur_node_features.clone()
                            cur_adj_features_tmp = cur_adj_features.clone()
                            cur_mol_size = rw_mol.GetNumAtoms()

                            traj_node_inputs['node_features'].pop(-1)
                            traj_node_inputs['adj_features'].pop(-1)
                            traj_node_inputs['node_features_cont'].pop(-1)
                            traj_node_inputs['rewards'].pop(-1)
                            traj_node_inputs['baseline_index'].pop(-1)
                            traj_node_inputs['node_cnt'].pop(-1)
                
                            ## pop adj
                            for _ in range(step_num_data_edge):
                                traj_adj_inputs['node_features'].pop(-1)
                                traj_adj_inputs['adj_features'].pop(-1)
                                traj_adj_inputs['edge_features_cont'].pop(-1)
                                traj_adj_inputs['index'].pop(-1)
                                traj_adj_inputs['rewards'].pop(-1)
                                traj_adj_inputs['baseline_index'].pop(-1)
                                traj_adj_inputs['edge_cnt'].pop(-1)
                            
                            mol_demon_edit = Chem.RWMol(rw_mol)
                            last_id2 = mol_demon_edit.AddAtom(Chem.Atom(6))
                            traj_node_inputs['node_features'].append(cur_node_features_tmp.clone())
                            traj_node_inputs['adj_features'].append(cur_adj_features_tmp.clone())
                            node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                            node_feature_cont[0, feature_id] = 1.0
                            traj_node_inputs['node_features_cont'].append(node_feature_cont)
                            traj_node_inputs['rewards'].append(torch.full(size=(1,1), fill_value=step_cnt).cuda())  # (1, 1)
                            traj_node_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1, 1)
                            traj_node_inputs['node_cnt'].append(torch.full(size=(1,), fill_value=float(i)).long().cuda())
                            
                            cur_node_features_tmp[0, last_id2, 0] = 1.0
                            cur_adj_features_tmp[0, :, last_id2, last_id2] = 1.0

                            flag_success = False
                            if cur_mol_size <= self.edge_unroll:
                                edge_total = cur_mol_size
                                start = 0
                                edge_idx = int(cur_mol_size * (cur_mol_size - 1) / 2)
                            else:
                                edge_total = self.edge_unroll
                                start = cur_mol_size - self.edge_unroll
                                edge_idx = int(self.edge_unroll * (self.edge_unroll - 1) / 2 + (cur_mol_size - self.edge_unroll) * self.edge_unroll)

                            if cur_mol_size > keep_size:
                                mol_demon_edit.AddBond(cur_mol_size-1, last_id2, Chem.rdchem.BondType.SINGLE)
                                valid = check_valency(mol_demon_edit)
                                if valid:
                                    flag_success = True
                                    for j in range(edge_total):
                                        edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                                        if j == edge_total - 1:
                                            edge_feature_cont[0, 0] = 1.0
                                        else:
                                            edge_feature_cont[0, 3] = 1.0
                                        
                                        traj_adj_inputs['node_features'].append(cur_node_features_tmp.clone())
                                        traj_adj_inputs['adj_features'].append(cur_adj_features_tmp.clone())
                                        traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)
                                        traj_adj_inputs['index'].append(torch.Tensor([[j + start, last_id2]]).long().cuda().view(1,-1))
                                        traj_adj_inputs['edge_cnt'].append(torch.full(size=(1,), fill_value=float(edge_idx)).long().cuda())
                                        traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                        traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)

                                        if j == edge_total - 1:
                                            cur_adj_features_tmp[0, 0, last_id2, j + start] = 1.0
                                            cur_adj_features_tmp[0, 0, j + start, last_id2] = 1.0
                                        else:
                                            cur_adj_features_tmp[0, 3, last_id2, j + start] = 1.0
                                            cur_adj_features_tmp[0, 3, j + start, last_id2] = 1.0

                                        edge_idx += 1
                                else:
                                    mol_demon_edit.RemoveBond(cur_mol_size-1, last_id2)

                                    count = 0
                                    while True:
                                        if count > 100:
                                            break
                                        if last_id2 > 12:
                                            k = np.random.randint(1, 13)
                                        else:
                                            k = np.random.randint(1, last_id2 + 1)
                                        if flag_reconstruct_from_node_adj:
                                            mol_demon_edit.AddBond(int(last_id2 - k), int(last_id2), Chem.rdchem.BondType.SINGLE)
                                        else:
                                            mol_demon_edit.AddBond(int(org_bfs_perm_origin[last_id2 - k]), int(last_id2), Chem.rdchem.BondType.SINGLE)

                                        valid = check_valency(mol_demon_edit)
                                        if valid:
                                            flag_success = True
                                            for j in range(edge_total):
                                                edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                                                if j + start == last_id2 - k:
                                                    edge_feature_cont[0, 0] = 1.0
                                                else:
                                                    edge_feature_cont[0, 3] = 1.0
                                                
                                                traj_adj_inputs['node_features'].append(cur_node_features_tmp.clone())
                                                traj_adj_inputs['adj_features'].append(cur_adj_features_tmp.clone())
                                                traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)
                                                traj_adj_inputs['index'].append(torch.Tensor([[j + start, last_id2 - k]]).long().cuda().view(1,-1))
                                                traj_adj_inputs['edge_cnt'].append(torch.full(size=(1,), fill_value=float(edge_idx)).long().cuda())
                                                traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                                traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)

                                                if j + start == last_id2 - k:
                                                    cur_adj_features_tmp[0, 0, last_id2, last_id2 - k] = 1.0
                                                    cur_adj_features_tmp[0, 0, last_id2 - k, last_id2] = 1.0
                                                else:
                                                    cur_adj_features_tmp[0, 3, last_id2, last_id2 - k] = 1.0
                                                    cur_adj_features_tmp[0, 3, last_id2 - k, last_id2] = 1.0

                                                edge_idx += 1
                                            break
                                        else:
                                            if flag_reconstruct_from_node_adj:
                                                mol_demon_edit.RemoveBond(int(last_id2 - k), int(last_id2))
                                            else:
                                                mol_demon_edit.RemoveBond(int(org_bfs_perm_origin[last_id2 - k]), int(last_id2))
                                            count += 1
                            else:
                                count = 0
                                while True:
                                    if count > 100:
                                        break
                                    if keep_size > 12:
                                        k = np.random.randint(1, 13)
                                    else:
                                        k = np.random.randint(1, keep_size+1)
                                    if flag_reconstruct_from_node_adj:
                                        mol_demon_edit.AddBond(int(keep_size-k), int(keep_size), Chem.rdchem.BondType.SINGLE)
                                    else:
                                        mol_demon_edit.AddBond(int(org_bfs_perm_origin[keep_size-k]), int(keep_size), Chem.rdchem.BondType.SINGLE)
                                    cur_adj_features_tmp[0, 0, keep_size - k, keep_size] = 1.0
                                    cur_adj_features_tmp[0, 0, keep_size, keep_size - k] = 1.0

                                    valid = check_valency(mol_demon_edit)
                                    if valid:
                                        flag_success = True
                                        for j in range(edge_total):
                                            edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                                            if j + start == last_id2 - k:
                                                edge_feature_cont[0, 0] = 1.0
                                            else:
                                                edge_feature_cont[0, 3] = 1.0
                                            
                                            traj_adj_inputs['node_features'].append(cur_node_features_tmp.clone())
                                            traj_adj_inputs['adj_features'].append(cur_adj_features_tmp.clone())
                                            traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)
                                            traj_adj_inputs['index'].append(torch.Tensor([[j + start, last_id2 - k]]).long().cuda().view(1,-1))
                                            traj_adj_inputs['edge_cnt'].append(torch.full(size=(1,), fill_value=float(edge_idx)).long().cuda())
                                            traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                            traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)

                                            if j + start == last_id2 - k:
                                                cur_adj_features_tmp[0, 0, last_id2, last_id2 - k] = 1.0
                                                cur_adj_features_tmp[0, 0, last_id2 - k, last_id2] = 1.0
                                            else:
                                                cur_adj_features_tmp[0, 3, last_id2, last_id2 - k] = 1.0
                                                cur_adj_features_tmp[0, 3, last_id2 - k, last_id2] = 1.0

                                            edge_idx += 1
                                        break
                                    else:
                                        if flag_reconstruct_from_node_adj:
                                            mol_demon_edit.RemoveBond(int(keep_size-k), int(keep_size))
                                        else:
                                            mol_demon_edit.RemoveBond(int(org_bfs_perm_origin[keep_size-k]), int(keep_size))
                                        count += 1

                            if flag_success and check_chemical_validity(mol_demon_edit): # successfully take one min action.
                                rw_mol = Chem.RWMol(mol_demon_edit)
                                cur_node_features = cur_node_features_tmp.clone()
                                cur_adj_features = cur_adj_features_tmp.clone()
                                is_continue = True
                                mol = rw_mol.GetMol()
                                node_features_each_iter_backup = cur_node_features.clone() # update node backup since new node is valid
                                adj_features_each_iter_backup = cur_adj_features.clone() #
                                added_num += 1                                        
                            else:
                                traj_node_inputs['node_features'].pop(-1)
                                traj_node_inputs['adj_features'].pop(-1)
                                traj_node_inputs['node_features_cont'].pop(-1)
                                traj_node_inputs['rewards'].pop(-1)
                                traj_node_inputs['baseline_index'].pop(-1)
                                traj_node_inputs['node_cnt'].pop(-1)
                                step_cnt += 1
                                continue
                    step_cnt += 1

                batch_length += 1

                #TODO: check the last iter of generation
                #(Thinking)
                # The last node was not added. So after we generate the second to last
                # node and all its edges, the rest adjacent matrix and node features should all be zero
                # But current implementation append
                num_atoms = mol.GetNumAtoms()
                assert num_atoms <= max_size_rl

                if num_atoms < max_size_rl:   
                    #! this implementation is buggy. we only mask the last node feature cont
                    #! But we ignore the non-zero node features in generating edges
                    #! this pattern will make model not to generated any edges between
                    #! the new-generated isolated node and exsiting subgraph.
                    #! this may be the biggest bug in Reinforce algorithm!!!!!
                    #! since the final iter/(step) has largest reward....!!!!!!!
                    #! work around1: add a counter and mask out all node feautres in generating edges of last iter.
                    #! work around2: do not append any data if the isolated node is not connected to subgraph.
                    # currently use work around2

                    # pop all the reinforce train-data add by at the generating the last isolated node and its edge
                    ## pop node
                    try:
                        traj_node_inputs['node_features'].pop(-1)
                        traj_node_inputs['adj_features'].pop(-1)
                        traj_node_inputs['node_features_cont'].pop(-1)
                        traj_node_inputs['rewards'].pop(-1)
                        traj_node_inputs['baseline_index'].pop(-1)
                        traj_node_inputs['node_cnt'].pop(-1)
                   
                        ## pop adj
                        for pop_cnt in range(step_num_data_edge):
                            traj_adj_inputs['node_features'].pop(-1)
                            traj_adj_inputs['adj_features'].pop(-1)
                            traj_adj_inputs['edge_features_cont'].pop(-1)
                            traj_adj_inputs['index'].pop(-1)
                            traj_adj_inputs['rewards'].pop(-1)
                            traj_adj_inputs['baseline_index'].pop(-1)
                            traj_adj_inputs['edge_cnt'].pop(-1)
                    except:
                        print('pop from empty list, take min action fail.')

                reward_valid = 2
                reward_property = 0
                reward_length = 0 

                assert mol is not None, 'mol is None...'
                final_valid = check_chemical_validity(mol)
                s_tmp = Chem.MolToSmiles(mol, isomericSmiles=True)
                assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!, \
                                raw is %s, cur is %s' % (s_raw, s_tmp)

                if not final_valid:
                    reward_valid -= 5 
                else:
                    final_mol = convert_radical_electrons_to_hydrogens(mol)
                    s = Chem.MolToSmiles(final_mol, isomericSmiles=True)

                    final_mol = Chem.MolFromSmiles(s)
                    # mol filters with negative rewards
                    if not steric_strain_filter(final_mol):  # passes 3D conversion, no excessive strain
                        reward_valid -= 1 #TODO: check the magnitude of this reward.
                    if not zinc_molecule_filter(final_mol):  # does not contain any problematic functional groups
                        reward_valid -= 1
                    
                    try:
                        score = calculate_min_plogp(final_mol)
                        score_raw = calculate_min_plogp(org_mol_true_raw)
                        if self.conf_rl['reward_type'] == 'exp':
                            reward_property += (np.exp(score / self.conf_rl['exp_temperature']) - self.conf_rl['exp_bias']) 
                        elif self.conf_rl['reward_type'] == 'linear':
                            reward_property += (score * self.conf_rl['linear_coeff'])
                        elif self.conf_rl['reward_type'] == 'imp':
                            reward_property += (score - score_raw)
                    except:
                        print('generated mol does not pass qed/plogp')

                reward_final_total = reward_valid + reward_property + reward_length
                per_mol_reward.append(reward_final_total)
                per_mol_property_score.append(reward_property)
                reward_decay = self.conf_rl['reward_decay']

                if added_num > 0:
                    node_inputs['node_features'].append(torch.cat(traj_node_inputs['node_features'], dim=0)) #append tensor of shape (max_size_rl, max_size_rl, self.node_dim)
                    node_inputs['adj_features'].append(torch.cat(traj_node_inputs['adj_features'], dim=0)) # append tensor of shape (max_size_rl, bond_dim, max_size_rl, max_size_rl)
                    node_inputs['node_features_cont'].append(torch.cat(traj_node_inputs['node_features_cont'], dim=0)) # append tensor of shape (max_size_rl, 9)

                    traj_node_inputs_baseline_index = torch.cat(traj_node_inputs['baseline_index'], dim=0) #(max_size_rl)
                    traj_node_inputs_rewards = torch.cat(traj_node_inputs['rewards'], dim=0) # tensor of shape (max_size_rl, 1)
                    traj_node_inputs_rewards[traj_node_inputs_rewards > 0] = \
                        reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_node_inputs_rewards[traj_node_inputs_rewards > 0])
                    node_inputs['rewards'].append(traj_node_inputs_rewards)  # append tensor of shape (max_size_rl, 1)                
                    node_inputs['baseline_index'].append(traj_node_inputs_baseline_index)
                    node_inputs['node_cnt'].append(torch.cat(traj_node_inputs['node_cnt'], dim=0))

                    for ss in range(traj_node_inputs_rewards.size(0)):
                        reward_baseline[traj_node_inputs_baseline_index[ss]][0] += 1.0
                        reward_baseline[traj_node_inputs_baseline_index[ss]][1] += traj_node_inputs_rewards[ss][0]                
                    
                    adj_inputs['node_features'].append(torch.cat(traj_adj_inputs['node_features'], dim=0)) # (step, max_size_rl, self.node_dim)
                    adj_inputs['adj_features'].append(torch.cat(traj_adj_inputs['adj_features'], dim=0)) # (step, bond_dim, max_size_rl, max_size_rl)
                    adj_inputs['edge_features_cont'].append(torch.cat(traj_adj_inputs['edge_features_cont'], dim=0)) # (step, 4)
                    adj_inputs['index'].append(torch.cat(traj_adj_inputs['index'], dim=0)) # (step, 2)
                    adj_inputs['edge_cnt'].append(torch.cat(traj_adj_inputs['edge_cnt'], dim=0))

                    traj_adj_inputs_baseline_index = torch.cat(traj_adj_inputs['baseline_index'], dim=0) #(step)                
                    traj_adj_inputs_rewards = torch.cat(traj_adj_inputs['rewards'], dim=0)
                    traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0] = \
                        reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0])
                    adj_inputs['rewards'].append(traj_adj_inputs_rewards)
                    adj_inputs['baseline_index'].append(traj_adj_inputs_baseline_index)

                    for ss in range(traj_adj_inputs_rewards.size(0)):
                        reward_baseline[traj_adj_inputs_baseline_index[ss]][0] += 1.0
                        reward_baseline[traj_adj_inputs_baseline_index[ss]][1] += traj_adj_inputs_rewards[ss][0]

        self.flow_core.train()
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()

        for i in range(reward_baseline.size(0)):
            if reward_baseline[i, 0] == 0:
                reward_baseline[i, 0] += 1.

        reward_baseline_per_step = reward_baseline[:, 1] / reward_baseline[:, 0] # (max_size_rl, )

        if in_baseline is not None:
            assert in_baseline.size() == reward_baseline_per_step.size()
            reward_baseline_per_step = reward_baseline_per_step * (1. - self.conf_rl['moving_coeff']) + in_baseline * self.conf_rl['moving_coeff']

        node_inputs_node_features = torch.cat(node_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        node_inputs_adj_features = torch.cat(node_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        node_inputs_node_features_cont = torch.cat(node_inputs['node_features_cont'], dim=0) # (total_size, 9)
        node_inputs_rewards = torch.cat(node_inputs['rewards'], dim=0).view(-1) # (total_size,)
        node_inputs_baseline_index = torch.cat(node_inputs['baseline_index'], dim=0).long() # (total_size,)
        node_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=node_inputs_baseline_index) #(total_size, )
        node_inputs_node_cnts = torch.cat(node_inputs['node_cnt'], dim=0)

        adj_inputs_node_features = torch.cat(adj_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        adj_inputs_adj_features = torch.cat(adj_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        adj_inputs_edge_features_cont = torch.cat(adj_inputs['edge_features_cont'], dim=0) # (total_size, 4)
        adj_inputs_index = torch.cat(adj_inputs['index'], dim=0) # (total_size, 2)
        adj_inputs_rewards = torch.cat(adj_inputs['rewards'], dim=0).view(-1) # (total_size,)
        adj_inputs_baseline_index = torch.cat(adj_inputs['baseline_index'], dim=0).long() #(total_size,)
        adj_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=adj_inputs_baseline_index) #(total_size, )
        adj_inputs_edge_cnts = torch.cat(adj_inputs['edge_cnt'], dim=0)

        if self.dp:
            node_function = self.flow_core.module.forward_rl_node
            edge_function = self.flow_core.module.forward_rl_edge

            node_function_old = self.flow_core_old.module.forward_rl_node
            edge_function_old = self.flow_core_old.module.forward_rl_edge
        else:
            node_function = self.flow_core.forward_rl_node
            edge_function = self.flow_core.forward_rl_edge
            
            node_function_old = self.flow_core_old.forward_rl_node
            edge_function_old = self.flow_core_old.forward_rl_edge

        z_node, _ = node_function(node_inputs_node_features, node_inputs_adj_features,
                                            node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

        z_edge, _ = edge_function(adj_inputs_node_features, adj_inputs_adj_features,
                                            adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )


        with torch.no_grad():
            z_node_old, _ = node_function_old(node_inputs_node_features, node_inputs_adj_features,
                                                node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

            z_edge_old, _ = edge_function_old(adj_inputs_node_features, adj_inputs_adj_features,
                                                adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )

        node_total_length = z_node.size(0) * float(self.node_dim)
        edge_total_length = z_edge.size(0) * float(self.bond_dim)

        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        node_base_log_probs_sm_select = torch.index_select(node_base_log_probs_sm, dim=0, index=node_inputs_node_cnts)
        ll_node = torch.sum(z_node * node_base_log_probs_sm_select, dim=(-1,-2))
        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        edge_base_log_probs_sm_select = torch.index_select(edge_base_log_probs_sm, dim=0, index=adj_inputs_edge_cnts)
        ll_edge = torch.sum(z_edge * edge_base_log_probs_sm_select, dim=(-1,-2))
        
        node_base_log_probs_sm_old = torch.nn.functional.log_softmax(self.node_base_log_probs_old, dim=-1)
        node_base_log_probs_sm_old_select = torch.index_select(node_base_log_probs_sm_old, dim=0, index=node_inputs_node_cnts)
        ll_node_old = torch.sum(z_node_old * node_base_log_probs_sm_old_select, dim=(-1,-2))
        edge_base_log_probs_sm_old = torch.nn.functional.log_softmax(self.edge_base_log_probs_old, dim=-1)
        edge_base_log_probs_sm_old_select = torch.index_select(edge_base_log_probs_sm_old, dim=0, index=adj_inputs_edge_cnts)
        ll_edge_old = torch.sum(z_edge_old * edge_base_log_probs_sm_old_select, dim=(-1,-2))

        ratio_node = torch.exp((ll_node - ll_node_old.detach()).clamp(max=10., min=-10.))
        ratio_edge = torch.exp((ll_edge - ll_edge_old.detach()).clamp(max=10., min=-10.))        

        if torch.isinf(ratio_node).any():
            raise RuntimeError('ratio node has inf entries')
       
        if torch.isinf(ratio_edge).any():
            raise RuntimeError('ratio edge has inf entries')
        if self.conf_rl['reward_type'] == 'imp' and self.conf_rl['no_baseline']:
            advantage_node = node_inputs_rewards
            advantage_edge = adj_inputs_rewards
        else:
            advantage_node = (node_inputs_rewards - node_inputs_baseline)
            advantage_edge = (adj_inputs_rewards - adj_inputs_baseline)

        surr1_node = ratio_node * advantage_node
        surr2_node = torch.clamp(ratio_node, 1-0.2, 1+0.2) * advantage_node

        surr1_edge = ratio_edge * advantage_edge
        surr2_edge = torch.clamp(ratio_edge, 1-0.2, 1+0.2) * advantage_edge

        if torch.isnan(surr1_node).any():
            raise RuntimeError('surr1 node has NaN entries')
        if torch.isnan(surr2_node).any():
            raise RuntimeError('surr2 node has NaN entries')
        if torch.isnan(surr1_edge).any():
            raise RuntimeError('surr1 edge has NaN entries')
        if torch.isnan(surr2_edge).any():
            raise RuntimeError('surr2 edge has NaN entries')                       

        return -((torch.min(surr1_node, surr2_node).sum() + torch.min(surr1_edge, surr2_edge).sum()) / (node_total_length + edge_total_length) - 1.0), per_mol_reward, per_mol_property_score, reward_baseline_per_step
   

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=12):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12, calculated from zink250K data)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (max_edge_unroll))
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).byte()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).byte()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).byte()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).byte()        
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).byte()

        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node-1].clone()
                    adj_masks2[cnt_edge][i,i] = 1
                else:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge-1].clone()
                    adj_masks2[cnt_edge][i, start + j -1] = 1
                    adj_masks2[cnt_edge][start + j -1, i] = 1
                cnt += 1
                cnt_edge += 1
        assert cnt == num_masks, 'masks cnt wrong'
        assert cnt_node == max_node_unroll, 'node masks cnt wrong'
        assert cnt_edge == num_mask_edge, 'edge masks cnt wrong'

    
        cnt = 0
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
        
            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1
        assert cnt == num_mask_edge, 'edge mask initialize fail'


        for i in range(max_node_unroll):
            if i == 0:
                continue
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                start = i - max_edge_unroll
                end = i 
            flow_core_edge_masks[i][start:end] = 1

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)
        
        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks

    
    def dis_log_prob(self, z):
        x_deq, adj_deq = z
        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        ll_node = torch.sum(x_deq * node_base_log_probs_sm, dim=(-1,-2))
        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        ll_edge = torch.sum(adj_deq * edge_base_log_probs_sm, dim=(-1,-2))
        return -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length))