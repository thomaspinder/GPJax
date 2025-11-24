import typing as tp

from flax import nnx
import jax.tree_util as jtu
import numpyro
import numpyro.distributions as dist

from gpjax.parameters import (
    Parameter,
)


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
            # Check for attached prior
            numpyro_props = getattr(param, "numpyro_properties", {})
            prior = numpyro_props.get("prior")

        if prior is None:
            return param

        # Sample
        value = numpyro.sample(name, prior)

        # Update parameter
        return param.replace(value)

    graphdef, state = nnx.split(model)

    new_state = jtu.tree_map_with_path(
        _param_callback, state, is_leaf=lambda x: isinstance(x, Parameter)
    )

    return nnx.merge(graphdef, new_state)
