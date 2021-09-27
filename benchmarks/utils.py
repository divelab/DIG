import os


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)
