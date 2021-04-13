

class Generator():
    def train_rand_gen(self, loader, *args, **kwargs):
        raise NotImplementedError("The function train_rand_gen is not implemented!")
    
    def run_rand_gen(self, *args, **kwargs):
        raise NotImplementedError("The function run_rand_gen is not implemented!")

    def train_prop_optim(self, *args, **kwargs):
        raise NotImplementedError("The function train_prop_optim is not implemented!")
    
    def run_prop_optim(self, *args, **kwargs):
        raise NotImplementedError("The function run_prop_optim is not implemented!")
    
    def train_cons_optim(self, loader, *args, **kwargs):
        raise NotImplementedError("The function train_cons_optim is not implemented!")
    
    def run_cons_optim(self, loader, *args, **kwargs):
        raise NotImplementedError("The function run_cons_optim is not implemented!")