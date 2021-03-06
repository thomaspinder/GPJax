from copy import deepcopy
from typing import Tuple

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from multipledispatch import dispatch

from .types import Array


def I(n: int) -> Array:
    """
    Compute an n x n identity matrix.
    :param n: The size of of the matrix.
    :return: An n x n identity matrix.
    """
    return jnp.eye(n)


def concat_dictionaries(a: dict, b: dict) -> dict:
    """
    Append one dictionary below another. If duplicate keys exist, then the key-value pair of the second supplied
    dictionary will be used.
    """
    return {**a, **b}


def merge_dictionaries(base_dict: dict, in_dict: dict) -> dict:
    """
    This will return a complete dictionary based on the keys of the first matrix. If the same key should exist in the
    second matrix, then the key-value pair from the first dictionary will be overwritten. The purpose of this is that
    the base_dict will be a complete dictionary of values such that an incomplete second dictionary can be used to
    update specific key-value pairs.

    :param base_dict: Complete dictionary of key-value pairs.
    :param in_dict: Subset of key-values pairs such that values from this dictionary will take precedent.
    :return: A merged single dictionary.
    """
    for k, v in base_dict.items():
        if k in in_dict.keys():
            base_dict[k] = in_dict[k]
    return base_dict


def sort_dictionary(base_dict: dict) -> dict:
    """
    Sort a dictionary based on the dictionary's key values.

    :param base_dict: The unsorted dictionary.
    :return: A dictionary sorted alphabetically on the dictionary's keys.
    """
    return dict(sorted(base_dict.items()))


@dispatch(jnp.DeviceArray)
def standardise(x: jnp.DeviceArray) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
    """
    Standardise a given matrix such that values are distributed according to a unit normal random variable. This is
    primarily designed for standardising a training dataset.

    :param x: A matrix of unstandardised values
    :return: A matrix of standardised values
    """
    xmean = jnp.mean(x, axis=0)
    xstd = jnp.std(x, axis=0)
    return (x - xmean) / xstd, xmean, xstd


@dispatch(jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray)
def standardise(
    x: jnp.DeviceArray, xmean: jnp.DeviceArray, xstd: jnp.DeviceArray
) -> jnp.DeviceArray:
    """
    Standardise a given matrix with respect to a given mean and standard deviation. This is primarily designed for
    standardising a test set of data with respect to the training data.

    :param x: A matrix of unstandardised values
    :param xmean: A precomputed mean vector
    :param xstd: A precomputed standard deviation vector
    :return: A matrix of standardised values
    """
    return (x - xmean) / xstd


def unstandardise(
    x: jnp.DeviceArray, xmean: jnp.DeviceArray, xstd: jnp.DeviceArray
) -> jnp.DeviceArray:
    """
    Unstandardise a given matrix with respect to a previously computed mean and standard deviation. This is designed
    for remapping a matrix back onto its original scale.

    :param x: A standardised matrix.
    :param xmean: A mean vector.
    :param xstd: A standard deviation vector.
    :return: A matrix of unstandardised values.
    """
    return (x * xstd) + xmean


def as_constant(parameter_set: dict, params: list) -> Tuple[dict, dict]:
    base_params = deepcopy(parameter_set)
    sparams = {}
    for param in params:
        sparams[param] = base_params[param]
        del base_params[param]
    return base_params, sparams
