from flax import nnx
import jax.tree_util as jtu
import numpyro
import numpyro.distributions as dist

from gpjax.parameters import Parameter


def tree_path_to_name(path: jtu.KeyPath, prefix: str = "") -> str:
    """Convert a JAX tree path to a dotted parameter name.

    As an example, the lengthscale parameter of an RBF kernel that was instantiated
    with the name "kernel" would then be registered with the name "kernel.lengthscale".

    Args:
        path: A JAX tree path (sequence of path keys).
        prefix: Optional prefix to prepend to the name.

    Returns:
        A dotted string representing the parameter name.
    """
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
    return f"{prefix}.{name}" if prefix else name


def resolve_prior(
    name: str,
    param: Parameter,
    priors: dict[str, dist.Distribution],
) -> dist.Distribution | None:
    """Resolve the prior precedence of a parameter.

    Explicit priors in the dictionary take precedence over attached priors. This step
    allows for explicit prior specification in the model definition, and then overriding
    with a different prior during inference.

    Args:
        name: The parameter name.
        param: The Parameter instance.
        priors: Dictionary mapping parameter names to distributions.

    Returns:
        The resolved distribution, or None if no prior is found.
    """
    prior = priors.get(name)
    if prior is None:
        numpyro_props = getattr(param, "numpyro_properties", {})
        prior = numpyro_props.get("prior")
    return prior


def register_parameters(
    model: nnx.Module,
    priors: dict[str, dist.Distribution] | None = None,
    prefix: str = "",
) -> nnx.Module:
    """
    Register GPJax parameters with Numpyro.

    Args:
        model: The GPJax model that contains parameters and is a subclass of nnx.Module.
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

        name = tree_path_to_name(path, prefix)
        prior = resolve_prior(name, param, priors)

        if prior is None:
            return param

        value = numpyro.sample(name, prior)
        return param.replace(value)

    graphdef, state = nnx.split(model)

    new_state = jtu.tree_map_with_path(
        _param_callback, state, is_leaf=lambda x: isinstance(x, Parameter)
    )

    return nnx.merge(graphdef, new_state)
