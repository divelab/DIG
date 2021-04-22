

class Generator():
    r"""
    The method base class for graph generation. To write a new graph generation method, create a new class
    inheriting from this class and implement the functions.
    """

    def train_rand_gen(self, loader, *args, **kwargs):
        r"""
        Running training for random generation task.
        """

        raise NotImplementedError("The function train_rand_gen is not implemented!")
    
    def run_rand_gen(self, *args, **kwargs):
        r"""
        Running graph generation for random generation task.
        """

        raise NotImplementedError("The function run_rand_gen is not implemented!")

    def train_prop_optim(self, *args, **kwargs):
        r"""
        Running training for property optimization task.
        """

        raise NotImplementedError("The function train_prop_optim is not implemented!")
    
    def run_prop_optim(self, *args, **kwargs):
        r"""
        Running graph generation for property optimization task.
        """

        raise NotImplementedError("The function run_prop_optim is not implemented!")
    
    def train_cons_optim(self, loader, *args, **kwargs):
        r"""
        Running training for constrained optimization task.
        """

        raise NotImplementedError("The function train_cons_optim is not implemented!")
    
    def run_cons_optim(self, loader, *args, **kwargs):
        r"""
        Running molecule optimization for constrained optimization task.
        """

        raise NotImplementedError("The function run_cons_optim is not implemented!")