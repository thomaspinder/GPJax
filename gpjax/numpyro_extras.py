import equinox as eqx
import jax.tree_util as jtu
import numpyro
import numpyro.distributions as dist
from paramax import AbstractUnwrappable


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
    param: AbstractUnwrappable,
    priors: dict[str, dist.Distribution],
) -> dist.Distribution | None:
    """Resolve the prior precedence of a parameter.

    Explicit priors in the dictionary take precedence over attached priors. This step
    allows for explicit prior specification in the model definition, and then overriding
    with a different prior during inference.

    Args:
        name: The parameter name.
        param: The AbstractUnwrappable instance.
        priors: Dictionary mapping parameter names to distributions.

    Returns:
        The resolved distribution, or None if no prior is found.
    """
    prior = priors.get(name)
    if prior is None:
        # Check if the parameter has a .prior attribute (our paramax parameter types do)
        prior = getattr(param, "prior", None)
    return prior


def register_parameters(
    model: eqx.Module,
    priors: dict[str, dist.Distribution] | None = None,
    prefix: str = "",
) -> eqx.Module:
    """
    Register GPJax parameters with Numpyro.

    This function walks the model's pytree, finds AbstractUnwrappable nodes,
    and registers them as NumPyro sample sites with the appropriate priors.

    Because AbstractUnwrappable instances are themselves eqx.Module subclasses,
    standard jtu.tree_flatten flattens through them to raw arrays. We use
    ``is_leaf=lambda x: isinstance(x, AbstractUnwrappable)`` to stop flattening
    at parameter boundaries and detect them as leaves.

    Args:
        model: The GPJax model that contains parameters and is a subclass of eqx.Module.
        priors: Optional dictionary mapping parameter names to Numpyro distributions.
        prefix: Optional prefix for parameter names.

    Returns:
        The model with parameters updated from Numpyro samples.
    """
    from gpjax.parameters import Real

    if priors is None:
        priors = {}

    def _is_leaf(x):
        return isinstance(x, AbstractUnwrappable)

    # Flatten with AbstractUnwrappable as leaf boundary
    paths_and_leaves = jtu.tree_flatten_with_path(model, is_leaf=_is_leaf)[0]
    _leaves, treedef = jtu.tree_flatten(model, is_leaf=_is_leaf)

    # Track already-seen parameter ids to handle shared references
    seen_ids: dict[int, str] = {}  # id -> sample site name

    new_leaves = []
    for path, leaf in paths_and_leaves:
        if not isinstance(leaf, AbstractUnwrappable):
            new_leaves.append(leaf)
            continue

        # Handle shared parameters: if we've already sampled this exact object,
        # reuse the same sampled value (via deterministic site or cached value).
        leaf_id = id(leaf)
        name = tree_path_to_name(path, prefix)

        if leaf_id in seen_ids:
            # Shared parameter -- skip (the first occurrence's replacement
            # will be used for all references because tree_unflatten preserves
            # the identity of repeated leaves).
            # We need to append the SAME Real wrapper as the first time.
            # Since jtu.tree_unflatten doesn't preserve identity for distinct objects,
            # we need to sample only once and reuse the Real wrapper.
            new_leaves.append(
                new_leaves[
                    _first_occurrence_index(
                        seen_ids[leaf_id], paths_and_leaves, prefix, _is_leaf
                    )
                ]
            )
            continue

        prior = resolve_prior(name, leaf, priors)

        if prior is None:
            new_leaves.append(leaf)
            seen_ids[leaf_id] = name
            continue

        value = numpyro.sample(name, prior)
        new_leaf = Real(value)
        new_leaves.append(new_leaf)
        seen_ids[leaf_id] = name

    return jtu.tree_unflatten(treedef, new_leaves)


def _first_occurrence_index(name, paths_and_leaves, prefix, is_leaf):
    """Find the index of the first occurrence of a parameter by name."""
    for i, (path, _leaf) in enumerate(paths_and_leaves):
        if tree_path_to_name(path, prefix) == name:
            return i
    raise ValueError(f"Could not find first occurrence of {name}")
