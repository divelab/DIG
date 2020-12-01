"""
Configuration file
"""

conf = {}
conf['model'] = 'schnet'
conf['task_type'] = 'regression'#'classification'
conf['metric'] = 'mae'#'prc'
conf['num_tasks'] = 1
conf['graph_level_feature'] = False

######################################################################################################################
# Settings for training
##    'epochs': maximum training epochs
##    'early_stopping': patience used to stop training
##    'lr': starting learning rate
##    'lr_decay_factor': learning rate decay factor
##    'lr_decay_step_size': step size of learning rate decay 
##    'dropout': dropout rate
##    'weight_decay': l2 regularizer term
##    'depth': number of layers
##    'batch_size': training batch_size
######################################################################################################################
conf['epochs'] = 100
conf['early_stopping'] = 50
conf['lr'] = 0.0005   ### [0.0001, 0.001]
conf['lr_decay_factor'] = 0.5 ### {0.5, 0.8}
conf['lr_decay_step_size'] = 50 ### {50, 80}
conf['dropout'] = 0 ### [0, 0.5]
conf['weight_decay'] = 0.00005 ### [0, 0.0005]
conf['batch_size'] = 64  ### [64, 1024]

######################################################################################################################
# Settings for val/test
##    'vt_batch_size': val/test batch_size
######################################################################################################################
conf['vt_batch_size'] = 1000

# schnet
conf['schnet_hidden_channels'] = 128
conf['num_filters'] = 128
conf['num_interactions'] = 6
conf['num_gaussians'] = 50
conf['schnet_cutoff'] = 10.0
conf['readout'] = 'add'

# dimenet
conf['dimenet_hidden_channels'] = 128
conf['num_blocks'] = 6
conf['num_bilinear'] = 8
conf['num_spherical'] = 7
conf['num_radial'] = 6
conf['dimenet_cutoff'] = 5.0
conf['envelope_exponent'] = 5
conf['num_before_skip'] = 1
conf['num_after_skip'] = 2
conf['num_output_layers'] = 3