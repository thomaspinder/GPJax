import objax
from .gps import Prior
from .gps.posteriors import Posterior
from typing import Union
GP = Union[Prior, Posterior]


def save(gp: GP, model_name: str, directory: str = './'):
    """
    Locally save the hyperparameters of a Gaussian process prior or posterior object.

    Args:
        gp: The GP object to be saved.
        model_name: The name under which the object should be saved.
        directory: The location where the GP should be stored.
    """
    if not model_name.endswith('.npz'):
        model_name += '.npz'
    objax.io.save_var_collection(f'{directory}{model_name}', gp.vars())
    print(
        f"Model successfully saved to {directory} under filename {model_name}")


def load(gp: GP, location: str):
    """
    Load the hyperparameters of a GP back into the a previously instantiated GP prior or posterior.

    Args:
        gp: The GP for which the hyperparameters should be stored.
        location: The local filepath where the .npz file exists
    """
    assert location.endswith(
        '.npz'), "Loaded variables should be under a .npz extension"
    objax.io.load_var_collection(location, gp.vars())
    return gp
