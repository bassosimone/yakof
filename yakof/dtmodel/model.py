"""
Model class
===========

Internal module defining the model class.

Please, prefer using the `model` package directly.
"""

from __future__ import annotations

import numbers

from functools import reduce
from typing import Iterator

import numpy as np
import pandas as pd

from scipy import interpolate, ndimage, stats

from ..symbols.constraint import Constraint, Distribution as ConstraintDistribution
from ..symbols.context_variable import ContextVariable
from ..symbols.index import Index, Distribution as IndexDistribution
from ..symbols.presence_variable import PresenceVariable

from ...numpybackend import evaluator
from ...symbolic.symbol import Symbol

Ensemble = Iterator[tuple[float, dict[ContextVariable, Symbol]]]
"""Type alias for specifying the expected ensemble type."""


class Model:
    def __init__(
        self,
        name,
        cvs: list[ContextVariable],
        pvs: list[PresenceVariable],
        indexes: list[Index],
        capacities: list[Index],
        constraints: list[Constraint],
    ) -> None:
        self.name = name
        self.cvs = cvs
        self.pvs = pvs
        self.indexes = indexes
        self.capacities = capacities
        self.constraints = constraints
        self.grid = None
        self.field = None
        self.field_elements = None
        self.index_vals = None

    def reset(self):
        self.grid = None
        self.field = None
        self.field_elements = None
        self.index_vals = None

    def evaluate(
        self,
        grid: dict[PresenceVariable, np.ndarray],
        ensemble: Ensemble,
    ):
        # 1. Figure out shapes and ensemble weights and values
        assert self.grid is None
        c_weight = np.array([c[0] for c in ensemble])
        c_values = pd.DataFrame([c[1] for c in ensemble])
        c_size = c_values.shape[0]

        # 2. Create the bindings
        bindings: evaluator.Bindings = {pv.name: grid[pv] for pv in self.pvs}

        # 3. Evaluate the indexes
        c_subs = {}
        state = evaluator.StateWithCache(bindings)
        for index in self.indexes:
            if index.cvs is None:
                if isinstance(index.value, numbers.Number):
                    c_subs[index] = [index.value] * c_size
                else:
                    assert isinstance(index.value, IndexDistribution)
                    c_subs[index] = index.value.rvs(size=c_size)
            else:
                assert isinstance(index.value, Symbol)
                args = [c_values[cv].values for cv in index.cvs]
                c_subs[index] = evaluator.evaluate(index.value.node, state)

        # 4. Ensure the indexes have the right shape
        grid_shape = (grid[self.pvs[0]].size, grid[self.pvs[1]].size)
        field = np.ones(grid_shape)
        field_elements = {}
        assert len(self.pvs) == 2  # TODO: generalize
        p_values = [
            np.expand_dims(grid[pv], axis=(i, 2)) for i, pv in enumerate(self.pvs)
        ]
        c_values = [
            np.expand_dims(c_subs[index], axis=(0, 1)) for index in self.indexes
        ]

        # 5. Evaluate the constraints
        for constraint in self.constraints:
            usage = evaluator.evaluate(constraint.usage.node, state)
            capacity = constraint.capacity
            # TODO: model type in declaration
            if not isinstance(capacity, ConstraintDistribution):
                unscaled_result = usage <= evaluator.evaluate(capacity.node, state)
            else:
                unscaled_result = 1.0 - capacity.cdf(usage)
            result = np.broadcast_to(np.dot(unscaled_result, c_weight), grid_shape)
            field_elements[constraint] = result
            field *= result

        # 6. Save the computation results
        self.index_vals = c_subs
        self.grid = grid
        self.field = field
        self.field_elements = field_elements
        return self.field

    # Note(bassosimone): from this point on, I am going to assume there is
    # nothing that we need to do in terms of symbolic computation. Therefore,
    # I am just going to use `type: ignore` to focus on what is above here.

    def get_index_value(self, i: Index) -> float:
        assert self.index_vals is not None
        return self.index_vals[i]

    def get_index_mean_value(self, i: Index) -> float:
        assert self.index_vals is not None
        return np.average(self.index_vals[i])

    def compute_sustainable_area(self) -> float:
        assert self.grid is not None
        grid = self.grid
        field = self.field

        return field.sum() * reduce(  # type: ignore
            lambda x, y: x * y,
            [axis.max() / (axis.size - 1) + 1 for axis in list(grid.values())],
        )

    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        assert self.grid is not None
        grid = self.grid
        field = self.field
        # TODO: fill value
        index = interpolate.interpn(
            grid.values(),
            field,
            np.array(presences),
            bounds_error=False,
            fill_value=0.0,
        )
        return np.mean(index)  # type: ignore

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        assert self.grid is not None
        grid = self.grid
        field_elements = self.field_elements
        # TODO: fill value
        indexes = {}
        for c in self.constraints:
            index = interpolate.interpn(
                grid.values(),
                field_elements[c],  # type: ignore
                np.array(presences),
                bounds_error=False,
                fill_value=0.0,
            )
            indexes[c] = np.mean(index)  # type: ignore
        return indexes

    def compute_modal_line_per_constraint(self) -> dict:
        assert self.grid is not None
        grid = self.grid
        field_elements = self.field_elements
        modal_lines = {}
        for c in self.constraints:
            fe = field_elements[c]  # type: ignore
            matrix = (fe <= 0.5) & (
                (ndimage.shift(fe, (0, 1)) > 0.5)
                | (ndimage.shift(fe, (0, -1)) > 0.5)
                | (ndimage.shift(fe, (1, 0)) > 0.5)
                | (ndimage.shift(fe, (-1, 0)) > 0.5)
            )
            (yi, xi) = np.nonzero(matrix)

            # TODO: decide whether two regressions are really necessary
            horizontal_regr = None
            vertical_regr = None
            try:
                horizontal_regr = stats.linregress(
                    grid[self.pvs[0]][xi], grid[self.pvs[1]][yi]
                )
            except ValueError:
                pass
            try:
                vertical_regr = stats.linregress(
                    grid[self.pvs[1]][yi], grid[self.pvs[0]][xi]
                )
            except ValueError:
                pass

            # TODO(pistore,bassosimone): find a better way to represent the lines (at the moment, we need to encode the endopoints
            # TODO(pistore,bassosimone): even before we implement the previous TODO, avoid hardcoding of line length (10000)

            def __vertical(regr) -> tuple[tuple[float, float], tuple[float, float]]:
                """Logic for computing the points with vertical regression"""
                if regr.slope != 0.00:
                    return ((regr.intercept, 0.0), (0.0, -regr.intercept / regr.slope))
                else:
                    return ((regr.intercept, regr.intercept), (0.0, 10000.0))

            def __horizontal(regr) -> tuple[tuple[float, float], tuple[float, float]]:
                """Logic for computing the points with horizontal regression"""
                if regr.slope != 0.0:
                    return ((0.0, -regr.intercept / regr.slope), (regr.intercept, 0.0))
                else:
                    return ((0.0, 10000.0), (regr.intercept, regr.intercept))

            if horizontal_regr and vertical_regr:
                # Use regression with better fit (higher rvalue)
                if horizontal_regr.rvalue < vertical_regr.rvalue:  # type: ignore
                    modal_lines[c] = __vertical(vertical_regr)
                else:
                    modal_lines[c] = __horizontal(horizontal_regr)

            elif horizontal_regr:
                modal_lines[c] = __horizontal(horizontal_regr)

            elif vertical_regr:
                modal_lines[c] = __vertical(vertical_regr)

            else:
                pass  # No regression is possible (eg median not intersecting the grid)

        return modal_lines

    def variation(self, new_name, *, change_indexes=None, change_capacities=None):
        # TODO: check if changes are valid (ie they change elements present in the model)
        if change_indexes is None:
            new_indexes = self.indexes
            change_indexes = {}
        else:
            new_indexes = []
            for index in self.indexes:
                if index in change_indexes:
                    new_indexes.append(change_indexes[index])
                else:
                    new_indexes.append(index)
        if change_capacities is None:
            new_capacities = self.capacities
            change_capacities = {}
        else:
            new_capacities = []
            for capacity in self.capacities:
                if capacity in change_capacities:
                    new_capacities.append(change_capacities[capacity])
                else:
                    new_capacities.append(capacity)
        new_constraints = []
        for constraint in self.constraints:
            new_constraints.append(
                Constraint(
                    constraint.usage.subs(change_indexes),  # type: ignore
                    constraint.capacity.subs(change_capacities),  # type: ignore
                    group=constraint.group,
                    name=constraint.name,
                )
            )
        return Model(
            new_name, self.cvs, self.pvs, new_indexes, new_capacities, new_constraints
        )
