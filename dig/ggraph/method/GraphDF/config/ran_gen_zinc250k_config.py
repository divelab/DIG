conf_data = {}
conf_data['num_max_node'] = 38
conf_data['num_bond_type'] = 3
conf_data['atom_list'] = [6, 7, 8, 9, 15, 16, 17, 35, 53] ## C, N, O, F, P, S, Cl, Br, I
conf_data['use_aug'] = True

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

conf_optim = {'lr': 0.001, 'weight_decay': 0, 'betas': (0.9, 0.999)}

conf = {}
conf['data'] = conf_data
conf['net'] = conf_net
conf['optim'] = conf_optim
conf['batch_size'] = 32
conf['ckpt_file'] = None
conf['epoches'] = 10
conf['verbose'] = 1
conf['save_epoch'] = 1
conf['num_gen'] = 100
conf['num_max_node_for_gen'] = 25
conf['num_min_node_for_gen'] = 7
conf['temperature_for_gen'] = [0.35, 0.2]