

class Generator():
    r"""
    The method base class for graph generation. To write a new graph generation method, create a new class
    inheriting from this class and implement the functions.
    """

    def train_rand_gen(self, loader, *args, **kwargs):
        r"""
        Running training for random generation task.

        Args:
            loader: The data loader for loading training samples.
        """

        raise NotImplementedError("The function train_rand_gen is not implemented!")
    
    def run_rand_gen(self, *args, **kwargs):
        r"""
        Running graph generation for random generation task.
        """

        raise NotImplementedError("The function run_rand_gen is not implemented!")

    def train_prop_opt(self, *args, **kwargs):
        r"""
        Running training for property optimization task.
        """

        raise NotImplementedError("The function train_prop_opt is not implemented!")
    
    def run_prop_opt(self, *args, **kwargs):
        r"""
        Running graph generation for property optimization task.
        """

        raise NotImplementedError("The function run_prop_opt is not implemented!")
    
    def train_const_prop_opt(self, loader, *args, **kwargs):
        r"""
        Running training for constrained optimization task.

        Args:
            loader: The data loader for loading training samples.
        """

        raise NotImplementedError("The function train_const_prop_opt is not implemented!")
    
    def run_const_prop_opt(self, *args, **kwargs):
        r"""
        Running molecule optimization for constrained optimization task.
        """

        raise NotImplementedError("The function run_const_prop_opt is not implemented!")