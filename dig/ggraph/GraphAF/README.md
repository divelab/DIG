abstract of re-implemented GraphAF

1. task one: density modeling
1.1. density modeling of QM9
1.2. density modeling of ZINC250K

2. task two: property optimization
2.1. property optimization of PlogP 
2.2. property optimization of QED

3. constrained property optimization
lowest plogp 800




Code for property optimization

Code structure:
1. model level --- all low level neural network implementations of GraphAF is under the folder model/, the implementation in the python scripts ended with '_rl.py' is for models used in property optimization, and '_con_rl.py' is for models used in constraint optimization.
2. task level --- Density modeling task is in dense_gen.py, property optimization task is in prop_optim.py, constraint optimization task is in con_optim.py. They implement the training, generation and evaluation stage of each task.
3. main --- main.py is the script to be runned. Configure your gpu id and output path here.

Before running main.py, configure your hyperparameters in the configuration files under the folder config/, dense_gen_config.py is for density generation, prop_optim_config.py is for property optimization and con_optim_config.py is for constraint optimization. The name of hyperparameters are tried to be consistent with those in GraphAF official code as much as possible. If you use a new configuration file, remember to include it properly in main.py.