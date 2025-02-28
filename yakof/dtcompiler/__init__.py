"""
Digital Twins Compiler
======================

....................
"""

from typing import Iterator, Protocol, Sequence, runtime_checkable

import numbers
import numpy as np
import pandas as pd

from ..frontend import abstract, bases, graph, spaces
from ..numpybackend import evaluator


@runtime_checkable
class Distribution(Protocol):
    """Protocol matching scipy.stats distributions interface"""

    def cdf(self, x: float | np.ndarray, *args, **kwds) -> float | np.ndarray: ...

    def rvs(self, size: int = 1) -> np.ndarray: ...


class ContextVariable:
    def __init__(self, name: str) -> None:
        self.__t = spaces.xyz.placeholder(name)

    @property
    def t(self) -> abstract.Tensor[bases.XYZ]:
        return self.__t

    @property
    def name(self) -> str:
        return self.__t.name


class PresenceVariable:
    def __init__(self, name: str, cvs: Sequence[ContextVariable]) -> None:
        self.__cvs = cvs
        self.__t = spaces.xyz.placeholder(name)

    @property
    def cvs(self) -> Iterator[ContextVariable]:
        return iter(self.__cvs)

    @property
    def t(self) -> abstract.Tensor[bases.XYZ]:
        return self.__t

    @property
    def name(self) -> str:
        return self.__t.name


class Index:
    def __init__(
        self,
        name: str = "",
        distribution: Distribution | None = None,
        value: graph.Scalar | abstract.Tensor[bases.XYZ] | None = None,
    ):
        self.__d: Distribution | None
        self.__t: abstract.Tensor[bases.XYZ]

        if value is None and distribution is None:
            raise ValueError("Either value or distribution must be provided")
        elif value is not None and distribution is not None:
            raise ValueError("Only one of value or distribution must be provided")
        elif value is not None:
            self.__t = (
                spaces.xyz.constant(value, name=name)
                if isinstance(value, graph.Scalar)
                else value
            )
            self.__d = None
        else:
            self.__t = spaces.xyz.placeholder(name)
            self.__d = distribution

    @property
    def distribution(self) -> Distribution | None:
        return self.__d

    @property
    def t(self) -> abstract.Tensor[bases.XYZ]:
        return self.__t

    @property
    def name(self) -> str:
        return self.__t.name


class Constraint:
    def __init__(self, usage: Index, capacity: Index) -> None:
        self.usage = usage
        self.capacity = capacity

    def _evaluate_usage(self, state: evaluator.State) -> np.ndarray:
        return evaluator.evaluate(self.usage.t.node, state)

    def _evaluate_constraint(
        self, state: evaluator.State, usage: np.ndarray
    ) -> np.ndarray:
        if self.capacity.distribution != None:
            return np.asarray(1.0) - self.capacity.distribution.cdf(usage)
        return usage <= evaluator.evaluate(self.capacity.t.node, state)

    def evaluate(self, state: evaluator.State) -> np.ndarray:
        return self._evaluate_constraint(state, self._evaluate_usage(state))


Ensemble = Iterator[tuple[float, dict[str, np.ndarray]]]
"""Type alias for specifying the expected ensemble type."""


class Model:
    def __init__(
        self,
        name,
        cvs: Sequence[ContextVariable],
        pvs: Sequence[PresenceVariable],
        indexes: Sequence[Index],
        capacities: Sequence[Index],
        constraints: Sequence[Constraint],
    ) -> None:
        self.name = name
        self.cvs = cvs
        self.pvs = pvs
        self.indexes = indexes
        self.capacities = capacities
        self.constraints = constraints

    def evaluate(
        self,
        grid: dict[PresenceVariable, np.ndarray],
        ensemble: Ensemble,
    ):
        # 1. Create empty state with empty bindings
        state = evaluator.StateWithCache({})

        # 2. Fill the placeholders for the presence variables
        if len(self.pvs) != 2:
            raise NotImplementedError("This model only supports 2D grids")
        state.set_placeholder_value(
            self.pvs[0].name,
            np.expand_dims(grid[self.pvs[0]], axis=(1, 2)),  # X, y, z
        )
        state.set_placeholder_value(
            self.pvs[1].name,
            np.expand_dims(grid[self.pvs[1]], axis=(0, 2)),  # x, Y, z
        )
        x_size = grid[self.pvs[0]].shape[0]
        y_size = grid[self.pvs[1]].shape[0]

        # 3. Create Z-aligned tensors for the ensemble weights
        weights = np.array([c[0] for c in ensemble])
        weights = np.expand_dims(weights, axis=(0, 1))  # x, y, Z
        ensemble_size = weights.shape[2]

        # 4. Create Z-aligned placeholders for the ensemble values
        for _, entry in ensemble:
            for name, value in entry.items():
                value = np.expand_dims(value, axis=(0, 1))  # x, y, Z
                state.set_placeholder_value(name, value)

        # 5. Create placeholders for the context variables values
        # TODO(bassosimone): implement this functionality

        # 6. Create placeholders for the index depending on random variates
        for index in self.indexes:
            if index.distribution != None:
                value = index.distribution.rvs(size=ensemble_size)
                value = np.expand_dims(value, axis=(0, 1))  # x, y, Z
                state.set_placeholder_value(index.name, value)

        # 7. Evaluate the constraints
        field = np.ones((x_size, y_size, ensemble_size))
        for constraint in self.constraints:
            field *= constraint.evaluate(state) * weights

        # 8. project the constraints over X, Y space
        return np.sum(field, axis=2)
