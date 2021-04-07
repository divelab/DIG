conf_data = {}
conf_data['num_max_node'] = 38
conf_data['num_bond_type'] = 3
conf_data['use_aug'] = False
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
conf_rl['modify_size'] = 5
conf_rl['penalty'] = True
conf_rl['update_iters'] = 4
conf_rl['reward_type'] = 'imp'
conf_rl['reward_decay'] = 0.90
conf_rl['exp_temperature'] = 3.0
conf_rl['exp_bias'] = 4.0
conf_rl['linear_coeff'] = 1.0
conf_rl['moving_coeff'] = 0.99
conf_rl['no_baseline'] = True
conf_rl['repeat_time'] = 200
conf_rl['min_optim_time'] = 50
conf_rl['warm_up'] = 24


conf = {}
conf['data'] = conf_data
conf['net'] = conf_net
conf['optim'] = conf_optim
conf['rl'] = conf_rl
conf['batch_size'] = 16
conf['ckpt_file'] = None
conf['pretrain_model'] = './saved_ckpts/con_optim/pretrain_graphaf.pth'
conf['reinforce_iters'] = 200
conf['save_iters'] = 20
conf['epoches'] = 10
conf['num_gen'] = 100
conf['num_max_node_for_gen'] = 38
conf['temperature_for_gen'] = [1.0, 1.0]