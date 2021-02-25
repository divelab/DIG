conf_data = {}
conf_data['num_max_node'] = 48
conf_data['num_bond_type'] = 3
conf_data['atom_list'] = [6, 7, 8, 9, 15, 16, 17, 35, 53] ## C, N, O, F, P, S, Cl, Br, I

conf_net = {}
conf_net['max_size'] = conf_data['num_max_node']
conf_net['edge_unroll'] = min(12, conf_data['num_max_node'] - 1)
conf_net['node_dim'] = len(conf_data['atom_list'])
conf_net['bond_dim'] = conf_data['num_bond_type'] + 1
conf_net['num_flow_layer'] = 12
conf_net['num_rgcn_layer'] = 3
conf_net['nhid'] = 128
conf_net['nout'] = 128
conf_net['use_gpu'] = True

conf_optim = {'lr': 0.0001, 'weight_decay': 0, 'betas': (0.9, 0.999)}

conf_rl = {}
conf_rl['penalty'] = True
conf_rl['update_iters'] = 4
conf_rl['property_type'] = 'qed'
conf_rl['reward_type'] = 'linear'
conf_rl['not_save_demon'] = True
conf_rl['reward_decay'] = 0.90
conf_rl['exp_temperature'] = 3.0
conf_rl['exp_bias'] = 4.0
conf_rl['linear_coeff'] = 2.0
conf_rl['split_batch'] = False
conf_rl['moving_coeff'] = 0.99
conf_rl['warm_up'] = 0
conf_rl['no_baseline'] = True


conf = {}
conf['data'] = conf_data
conf['net'] = conf_net
conf['optim'] = conf_optim
conf['rl'] = conf_rl
conf['batch_size'] = 8
conf['ckpt_file'] = None
conf['pretrain_model'] = 'saved_ckpts/prop_optim/pretrain_qed.pth'
conf['reinforce_iters'] = 200
conf['save_iters'] = 20
conf['num_gen'] = 10
conf['num_max_node_for_gen'] = 48
conf['num_min_node_for_gen'] = 0
conf['temperature_for_gen'] = [0.8,0.1]