"""
Model Definition
================

This module defines the core Model class which represents a digital twins
sustainability assessment framework. The Model combines presence variables,
context variables, indices, capacities and constraints into an evaluable model
that can assess environmental constraints across a spatial grid while
incorporating uncertainty via ensemble methods.
"""

from typing import Sequence, cast

import numpy as np

from . import ensemble

from .constraint import Constraint, CumulativeDistribution
from .context import ContextVariable
from .geometry import Tensor
from .index import Index
from .presence import PresenceVariable

from ..frontend import graph, linearize
from ..numpybackend import executor


class Model:
    def __init__(
        self,
        name,
        cvs: Sequence[ContextVariable],
        pvs: Sequence[PresenceVariable],
        indexes: Sequence[Tensor],
        capacities: Sequence[Tensor],
        constraints: Sequence[Constraint],
    ) -> None:
        """Initialize the digital-twins model.

        Args:
            name: The name of the model
            cvs: Sequence of context variables representing uncertainty dimensions
            pvs: Sequence of presence variables defining the spatial grid
            indexes: Sequence of index tensors
            capacities: Sequence of capacity tensors
            constraints: Sequence of constraints to be evaluated
        """
        self.name = name
        self.cvs = cvs
        self.pvs = pvs
        self.indexes = indexes
        self.capacities = capacities
        self.constraints = constraints

    def __evaluate_constraint(
        self,
        constr: Constraint,
        cache: dict[graph.Node, np.ndarray],
    ) -> np.ndarray:
        """Evaluate a single constraint with the provided cache.

        Args:
            constr: The constraint to evaluate
            cache: Dictionary mapping nodes to their evaluated values

        Returns:
            A numpy array representing the constraint satisfaction field
        """
        # TODO(bassosimone): to some extent, this could also be a method of
        # the actual Constraint class, since the distribution already specifies
        # we're operating in the numpy domain. The same argument could
        # potentially hold for the evaluation of the indexes.
        usage = cache[constr.usage.node]
        if isinstance(constr.capacity, CumulativeDistribution):
            return np.asarray(1.0) - constr.capacity.cdf(usage)
        return usage <= cache[constr.capacity.node]

    def evaluate(
        self,
        grid: dict[PresenceVariable, np.ndarray],
        ensemble: ensemble.Iter,
    ) -> np.ndarray:
        """Evaluate the model over a grid and ensemble of scenarios.

        Produces a sustainability field that aggregates the constraint
        satisfaction across the ensemble of parameter settings.

        Args:
            grid: Dictionary mapping presence variables to grid coordinate arrays
            ensemble: Iterator over weighted ensemble scenarios

        Returns:
            A 2D numpy array representing the sustainability field

        Raises:
            NotImplementedError: If more than 2 presence variables are specified
        """
        # TODO(bassosimone): the original implementation is also saving the
        # value of each constraint, which is very useful for debugging. I am
        # a bit torn with respect to this. For now, I have not added this
        # because I wanted this class to be immutable. Need to think a bit more.

        # 1. Create empty state with empty bindings
        cache: dict[graph.Node, np.ndarray] = {}

        # 2. Fill the placeholders for the presence variables
        if len(self.pvs) != 2:
            raise NotImplementedError("This model only supports 2D grids")

        cache[self.pvs[0].node] = cast(
            np.ndarray,
            np.expand_dims(grid[self.pvs[0]], axis=(0, 2)),  # x, Y, z
        )
        y_size = grid[self.pvs[0]].shape[0]

        cache[self.pvs[1].node] = cast(
            np.ndarray, np.expand_dims(grid[self.pvs[1]], axis=(1, 2))  # X, y, z
        )
        x_size = grid[self.pvs[1]].shape[0]

        # 3. Create Z-aligned tensors for the ensemble weights
        weights = np.array([c[0] for c in ensemble])
        weights = np.expand_dims(weights, axis=(0, 1))  # x, y, Z
        ensemble_size = weights.shape[2]

        # 4. Create Z-aligned placeholders for the ensemble values
        collector: dict[ContextVariable, list[float]] = {}
        for _, entry in ensemble:
            for cv, value in entry.items():
                collector.setdefault(cv, []).append(value)

        for key, values in collector.items():
            values = np.asarray(values)
            values = np.expand_dims(values, axis=(0, 1))  # x, y, Z
            cache[key.node] = values

        # 5. Create placeholders for the indexes
        for index in self.indexes:
            if isinstance(index, Index):
                # TODO(bassosimone): keep in mind that random variates sampling
                # here is partially incorrect, because we're not extending the
                # ensemble space and draw "enough" samples from the distribution.
                value = index.distribution.rvs(size=ensemble_size)
                value = np.expand_dims(value, axis=(0, 1))  # x, y, Z
                cache[index.node] = value

        # 6. Build the list of graph nodes to evaluate
        allnodes: list[graph.Node] = []
        for constr in self.constraints:
            allnodes.append(constr.usage.node)
            if not isinstance(constr.capacity, CumulativeDistribution):
                allnodes.append(constr.capacity.node)

        # 7. Sort and evaluate the graph in topological order
        allnodes = linearize.forest(*allnodes)
        state = executor.State(values=cache, flags=graph.NODE_FLAG_TRACE)
        for node in allnodes:
            executor.evaluate(state, node)

        # 8. Compute the sustainability field based on the results
        field = np.ones((x_size, y_size, ensemble_size))
        for constr in self.constraints:
            field *= self.__evaluate_constraint(constr, cache)

        # 9. Apply the ensemble weights to the field
        field *= weights

        # 10. project the constraints over X, Y space by summing over Z
        return np.sum(field, axis=2)
