conf_data = {}
conf_data['num_max_node'] = 38 # maximum number of atoms 
conf_data['num_bond_type'] = 3 # number of types of bond, single, double and Triple
conf_data['atom_list'] = [6, 7, 8, 9, 15, 16, 17, 35, 53] ## C, N, O, F, P, S, Cl, Br, I

conf_net = {}
conf_net['max_size'] = conf_data['num_max_node']
conf_net['edge_unroll'] = min(12, conf_data['num_max_node'] - 1) # maximum incident edges of one node
conf_net['node_dim'] = len(conf_data['atom_list'])
conf_net['bond_dim'] = conf_data['num_bond_type'] + 1 # add one for virtual edge (no bond)
conf_net['num_flow_layer'] = 12
conf_net['num_rgcn_layer'] = 3
conf_net['nhid'] = 128
conf_net['nout'] = 128
conf_net['deq_coeff'] = 0.9
conf_net['st_type'] = 'exp'
conf_net['use_df'] = False
conf_net['use_gpu'] = True

conf_optim = {'lr': 0.0001, 'weight_decay': 0, 'betas': (0.9, 0.999)}

conf_rl = {}
conf_rl['penalty'] = True # if penalty reward for chemical valency is used
conf_rl['update_iters'] = 4 # the frequency to copy parameters from new model to old model in PPO
# conf_rl['property_type'] = 'plogp'
conf_rl['property_type'] = 'qed' # qed or plogp
# conf_rl['reward_type'] = 'exp'
conf_rl['reward_type'] = 'linear' # linear or exp
conf_rl['reward_decay'] = 0.97 # the decay factor \gamma used in reward
conf_rl['exp_temperature'] = 3.0 # temperature of exp reward
conf_rl['exp_bias'] = 4.0
conf_rl['linear_coeff'] = 2.0 # multiplicative constant of linear reward
conf_rl['split_batch'] = False
conf_rl['moving_coeff'] = 0.99 # momentum coefficient for computing advantage function (the V in Q - V) in PPO
conf_rl['warm_up'] = 24 # warm up iterations
conf_rl['no_baseline'] = False # if advantage function (Q - V) or value function (Q) are used in PPO
conf_rl['divide_loss'] = True


conf = {}
conf['data'] = conf_data
conf['net'] = conf_net
conf['optim'] = conf_optim
conf['rl'] = conf_rl
conf['batch_size'] = 64 # batch size, in rl finetune it is the number of molecules generated in each training iteration
conf['ckpt_file'] = None
conf['dense_gen_model'] = 'ckpt/dense_gen_net_10.pth' # the path to the pretrained model file
conf['reinforce_iters'] = 200
conf['valid_iters'] = 2
conf['verbose'] = 1
conf['save_ckpt'] = 1
# the number of molecules, maximum/minimum number of nodes and temperature to be used in validation
conf['num_gen'] = 100
conf['num_max_node_for_gen'] = 38
conf['num_min_node_for_gen'] = 10
conf['temperature_for_gen'] = 0.75