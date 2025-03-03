"""
........
"""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence, cast, runtime_checkable

import numpy as np
import random

from yakof.frontend import abstract, autoenum, bases, graph, linearize, spaces
from yakof.numpybackend import executor


@runtime_checkable
class Distribution(Protocol):
    """Protocol matching scipy.stats distributions interface"""

    def cdf(self, x: float | np.ndarray, *args, **kwds) -> float | np.ndarray: ...

    def rvs(self, size: int = 1) -> np.ndarray: ...


class Index(abstract.Tensor[bases.XYZ]):
    def __init__(
        self,
        name: str,
        value: float | int | bool | Distribution | graph.Node,
    ) -> None:
        self.distribution: Distribution | None
        if isinstance(value, (float, int, bool)):
            super().__init__(spaces.xyz, graph.constant(value, name=name))
            self.distribution = None
        elif isinstance(value, Distribution):
            super().__init__(spaces.xyz, graph.placeholder(name))
            self.distribution = value
        else:
            super().__init__(spaces.xyz, value)
            self.distribution = None


class ContextVariable(Index):
    pass


# TODO(bassosimone): shrink this thing a bit and merge with non-uniform
class UniformCategoricalContextVariable(ContextVariable):
    def __init__(self, name: str, values: Sequence[str]) -> None:
        self.__enum = autoenum.Type(spaces.z, name)
        super().__init__(name, self.__enum.tensor.node)
        self.__size = len(values)
        self.__values = values
        self.__mapping: dict[str, autoenum.Value[bases.Z]] = {}
        for value in values:
            self.__mapping[value] = autoenum.Value(self.__enum, value)

    def __eq__(self, value: abstract.Tensor[bases.Z] | str | float | int) -> abstract.Tensor[bases.Z]:  # type: ignore
        if isinstance(value, (float, int)):
            return spaces.z.equal(self.__enum.tensor, spaces.z.constant(value))
        if isinstance(value, str):
            return spaces.z.equal(self.__enum.tensor, self.__mapping[value].tensor)
        return spaces.z.equal(self.__enum.tensor, value)

    def __hash__(self) -> int:
        return id(self)

    def support_size(self) -> int:
        return self.__size

    def sample(
        self, nr: int = 1, *, subset: list | None = None, force_sample: bool = False
    ) -> list:
        # TODO: subset (if defined) should be a subset of the support (also: with repetitions?)

        print("ELLIOT", "sample", nr, subset, force_sample)

        (values, size) = (
            (self.__values, self.__size) if subset is None else (subset, len(subset))
        )

        if force_sample or nr < size:
            ratio = 1 / nr
            keys = random.choices(values, k=nr)
        else:
            ratio = 1 / size
            keys = values

        rv =  [(ratio, self.__mapping[k].value) for k in keys]
        print("ELLIOT", "rv", rv)
        return rv


class PresenceVariable(Index):
    def __init__(self, name: str, cvs: Sequence[ContextVariable]) -> None:
        super().__init__("", graph.placeholder(name))
        self.cvs = cvs


def _ensure_index(value: Index | abstract.Tensor[bases.XYZ]) -> Index:
    return Index("", value.node) if not isinstance(value, Index) else value


class Constraint:
    def __init__(
        self,
        usage: Index | abstract.Tensor[bases.XYZ],
        capacity: Index | abstract.Tensor[bases.XYZ],
    ) -> None:
        self.usage = _ensure_index(usage)
        self.capacity = _ensure_index(capacity)


EnsembleWeight = float

EnsembleVariables = dict[ContextVariable, float]

Ensemble = Iterator[tuple[EnsembleWeight, EnsembleVariables]]


class Model:
    def __init__(
        self,
        name,
        cvs: Sequence[ContextVariable],
        pvs: Sequence[PresenceVariable],
        indexes: Sequence[Index | abstract.Tensor[bases.XYZ]],
        capacities: Sequence[Index],
        constraints: Sequence[Constraint],
    ) -> None:
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
        usage = cache[constr.usage.node]
        if constr.capacity.distribution != None:
            return np.asarray(1.0) - constr.capacity.distribution.cdf(usage)
        return usage <= cache[constr.capacity.node]

    def evaluate(
        self,
        grid: dict[PresenceVariable, np.ndarray],
        ensemble: Ensemble,
        debugflags: int = 0,
    ) -> np.ndarray:
        # 1. Create empty state with empty bindings
        cache: dict[graph.Node, np.ndarray] = {}

        # 2. Fill the placeholders for the presence variables
        if len(self.pvs) != 2:
            raise NotImplementedError("This model only supports 2D grids")
        cache[self.pvs[0].node] = cast(
            np.ndarray,
            np.expand_dims(grid[self.pvs[0]], axis=(0, 2)),  # x, Y, z
        )
        cache[self.pvs[1].node] = cast(
            np.ndarray, np.expand_dims(grid[self.pvs[1]], axis=(1, 2))  # X, y, z
        )
        x_size = grid[self.pvs[0]].shape[0]
        y_size = grid[self.pvs[1]].shape[0]

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

        # 5. Create placeholders for the index depending on random variates
        for index in self.indexes:
            if isinstance(index, Index) and index.distribution != None:
                value = index.distribution.rvs(size=ensemble_size)
                value = np.expand_dims(value, axis=(0, 1))  # x, y, Z
                cache[index.node] = value

        # 6. Build the list of graph nodes to evaluate
        allnodes: list[graph.Node] = []
        for constr in self.constraints:
            allnodes.append(constr.usage.node)
            if constr.capacity.distribution == None:
                allnodes.append(constr.capacity.node)

        # 7. Sort and evaluate the graph in topological order
        allnodes = linearize.forest(*allnodes)
        state = executor.State(values=cache, flags=debugflags)
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
