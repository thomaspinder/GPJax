import typing as tp

from flax import nnx
import jax.tree_util as jtu
import numpyro
import numpyro.distributions as dist

from gpjax.parameters import (
    FillTriangularTransform,
    Parameter,
)


def _get_default_prior(tag, shape, ndim):
    if tag in ("positive", "non_negative"):
        return dist.LogNormal(0.0, 1.0).expand(shape).to_event(ndim)
    if tag == "real":
        return dist.Normal(0.0, 1.0).expand(shape).to_event(ndim)
    if tag == "sigmoid":
        return dist.Uniform(0.0, 1.0).expand(shape).to_event(ndim)
    if tag == "lower_triangular":
        N = shape[-1]
        K = N * (N + 1) // 2
        batch_shape = shape[:-2]
        base_shape = batch_shape + (K,)
        base_dist = dist.Normal(0.0, 1.0).expand(base_shape).to_event(1)
        td = dist.TransformedDistribution(base_dist, FillTriangularTransform())
        return td.to_event(len(batch_shape))
    return dist.Normal(0.0, 1.0).expand(shape).to_event(ndim)


def register_parameters(
    model: nnx.Module,
    priors: tp.Dict[str, dist.Distribution] | None = None,
    prefix: str = "",
) -> nnx.Module:
    """
    Register GPJax parameters with Numpyro.

    Args:
        model: The GPJax model (flax.nnx.Module).
        priors: Optional dictionary mapping parameter names to Numpyro distributions.
        prefix: Optional prefix for parameter names.

    Returns:
        The model with parameters updated from Numpyro samples.
    """
    if priors is None:
        priors = {}

    def _param_callback(path, param):
        if not isinstance(param, Parameter):
            return param

        # Construct name
        name_parts = []
        for p in path:
            if isinstance(p, jtu.DictKey):
                name_parts.append(str(p.key))
            elif isinstance(p, jtu.SequenceKey):
                name_parts.append(str(p.idx))
            elif isinstance(p, jtu.GetAttrKey):
                name_parts.append(str(p.name))
            else:
                name_parts.append(str(p))

        name = ".".join(name_parts)
        if prefix:
            name = f"{prefix}.{name}"

        # Determine prior
        prior = priors.get(name)
        if prior is None:
            prior = _get_default_prior(param.tag, param.value.shape, param.value.ndim)

        # Sample
        value = numpyro.sample(name, prior)

        # Update parameter
        return param.replace(value)

    graphdef, state = nnx.split(model)

    new_state = jtu.tree_map_with_path(
        _param_callback, state, is_leaf=lambda x: isinstance(x, Parameter)
    )

    return nnx.merge(graphdef, new_state)
