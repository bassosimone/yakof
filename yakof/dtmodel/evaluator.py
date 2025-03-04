from typing import cast

import numpy as np

from ..frontend import graph, linearize
from ..numpybackend import executor

from . import constraint, context, ensemble, index, model, presence


def _evaluate_constraint(
    constr: constraint.Expression,
    cachestate: dict[graph.Node, np.ndarray],
) -> np.ndarray:
    usage = cachestate[constr.usage.node]
    if isinstance(constr.capacity, constraint.CapacityDistribution):
        return np.asarray(1.0) - constr.capacity.cdf(usage)
    return usage <= cachestate[constr.capacity.node]


def evaluate(
    model: model.Model,
    grid: dict[presence.Variable, np.ndarray],
    ensemble: ensemble.Iterator,
) -> np.ndarray:
    # 1. Create empty state with empty bindings
    cache: dict[graph.Node, np.ndarray] = {}

    # 2. Fill the placeholders for the presence variables
    if len(model.pvs) != 2:
        raise NotImplementedError("This model only supports 2D grids")
    cache[model.pvs[0].node] = cast(
        np.ndarray,
        np.expand_dims(grid[model.pvs[0]], axis=(0, 2)),  # x, Y, z
    )
    cache[model.pvs[1].node] = cast(
        np.ndarray, np.expand_dims(grid[model.pvs[1]], axis=(1, 2))  # X, y, z
    )
    x_size = grid[model.pvs[0]].shape[0]
    y_size = grid[model.pvs[1]].shape[0]

    # 3. Create Z-aligned tensors for the ensemble weights
    weights = np.array([c[0] for c in ensemble])
    weights = np.expand_dims(weights, axis=(0, 1))  # x, y, Z
    ensemble_size = weights.shape[2]

    # 4. Create Z-aligned placeholders for the ensemble values
    collector: dict[context.Variable, list[float]] = {}
    for _, entry in ensemble:
        for cv, value in entry.items():
            collector.setdefault(cv, []).append(value)
    for key, values in collector.items():
        values = np.asarray(values)
        values = np.expand_dims(values, axis=(0, 1))  # x, y, Z
        cache[key.node] = values

    # 5. Create placeholders for the index depending on random variates
    for idx in model.indexes:
        if isinstance(idx, index.Placeholder):
            value = idx.initializer.rvs(size=ensemble_size)
            value = np.expand_dims(value, axis=(0, 1))  # x, y, Z
            cache[idx.node] = value

    # 6. Build the list of graph nodes to evaluate
    allnodes: list[graph.Node] = []
    for constr in model.constraints:
        allnodes.append(constr.usage.node)
        if not isinstance(constr.capacity, constraint.CapacityDistribution):
            allnodes.append(constr.capacity.node)

    # 7. Sort and evaluate the graph in topological order
    allnodes = linearize.forest(*allnodes)
    state = executor.State(values=cache, flags=graph.NODE_FLAG_TRACE)
    for node in allnodes:
        executor.evaluate(state, node)

    # 8. Compute the sustainability field based on the results
    field = np.ones((x_size, y_size, ensemble_size))
    for constr in model.constraints:
        field *= _evaluate_constraint(constr, cache)

    # 9. Apply the ensemble weights to the field
    field *= weights

    # 10. project the constraints over X, Y space by summing over Z
    return np.sum(field, axis=2)
