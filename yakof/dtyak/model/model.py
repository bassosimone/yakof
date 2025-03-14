from __future__ import annotations

import numbers

from functools import reduce

import numpy as np
import pandas as pd
from sympy import lambdify
from scipy import interpolate, ndimage, stats

from ..symbols.constraint import Constraint, CumulativeDistribution
from ..symbols.context_variable import ContextVariable
from ..symbols.index import Index, Sampleable
from ..symbols.presence_variable import PresenceVariable

from ...frontend import graph, linearize
from ...numpybackend import executor
from ...sympyke import symbol


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

    def evaluate(self, grid, ensemble):
        assert self.grid is None

        # [pre] extract the weights and the size of the ensemble
        c_weight = np.array([c[0] for c in ensemble])
        c_size = c_weight.shape[0]

        # [pre] create empty placeholders
        c_subs: dict[graph.Node, np.ndarray] = {}

        # [pre] add global unique symbols
        for entry in symbol.symbol_table.values():
            c_subs[entry.node] = np.array(entry.name)

        # [pre] add context variables
        collector: dict[ContextVariable, list[float]] = {}
        for _, entry in ensemble:
            for cv, value in entry.items():
                collector.setdefault(cv, []).append(value)
        for key, values in collector.items():
            c_subs[key.node] = np.asarray(values)

        # [pre] evaluate the indexes depending on distributions
        #
        # TODO(bassosimone): the size used here is too small
        for index in self.indexes + self.capacities:
            if isinstance(index.value, Sampleable):
                c_subs[index.node] = np.asarray(index.value.rvs(size=c_size))

        # [eval] expand dimensions for all values computed thus far
        for key in c_subs:
            c_subs[key] = np.expand_dims(c_subs[key], axis=(0, 1))

        # [eval] add presence variables and expand dimensions
        assert len(self.pvs) == 2  # TODO: generalize
        for i, pv in enumerate(self.pvs):
            c_subs[pv.node] = np.expand_dims(grid[pv], axis=(i, 2))

        # [eval] collect all the nodes to evaluate
        all_nodes: list[graph.Node] = []
        for constraint in self.constraints:
            all_nodes.append(constraint.usage)
            if not isinstance(constraint.capacity, CumulativeDistribution):
                all_nodes.append(constraint.capacity)
        for index in self.indexes + self.capacities:
            all_nodes.append(index.node)

        # [eval] actually evaluate all the nodes
        state = executor.State(c_subs, graph.NODE_FLAG_TRACE)
        for node in linearize.forest(*all_nodes):
            executor.evaluate(state, node)

        # [post] compute the sustainability field
        grid_shape = (grid[self.pvs[0]].size, grid[self.pvs[1]].size)
        field = np.ones(grid_shape)
        field_elements = {}
        for constraint in self.constraints:
            # Get usage
            usage = c_subs[constraint.usage]

            # Get capacity
            capacity = constraint.capacity
            if not isinstance(capacity, CumulativeDistribution):
                unscaled_result = usage <= c_subs[capacity]
            else:
                unscaled_result = 1.0 - capacity.cdf(usage)

            # Apply weights and store the result
            result = np.broadcast_to(np.dot(unscaled_result, c_weight), grid_shape)
            field_elements[constraint] = result
            field *= result

        # [post] store the results
        self.index_vals = c_subs
        self.grid = grid
        self.field = field
        self.field_elements = field_elements
        return self.field

    def get_index_value(self, i: Index) -> float:
        assert self.index_vals is not None
        return self.index_vals[i.node]

    def get_index_mean_value(self, i: Index) -> float:
        assert self.index_vals is not None
        return np.average(self.index_vals[i.node])

    def compute_sustainable_area(self) -> float:
        assert self.grid is not None
        assert self.field is not None
        grid = self.grid
        field = self.field

        return field.sum() * reduce(
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
        return np.mean(index)

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        assert self.grid is not None
        assert self.field_elements is not None
        grid = self.grid
        field_elements = self.field_elements
        # TODO: fill value
        indexes = {}
        for c in self.constraints:
            index = interpolate.interpn(
                grid.values(),
                field_elements[c],
                np.array(presences),
                bounds_error=False,
                fill_value=0.0,
            )
            indexes[c] = np.mean(index)
        return indexes

    def compute_modal_line_per_constraint(self) -> dict:
        assert self.grid is not None
        assert self.field_elements is not None
        grid = self.grid
        field_elements = self.field_elements
        modal_lines = {}
        for c in self.constraints:
            fe = field_elements[c]
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

            def _vertical(regr) -> tuple[tuple[float, float], tuple[float, float]]:
                """Logic for computing the points with vertical regression"""
                if regr.slope != 0.00:
                    return ((regr.intercept, 0.0), (0.0, -regr.intercept / regr.slope))
                else:
                    return ((regr.intercept, regr.intercept), (0.0, 10000.0))

            def _horizontal(regr) -> tuple[tuple[float, float], tuple[float, float]]:
                """Logic for computing the points with horizontal regression"""
                if regr.slope != 0.0:
                    return ((0.0, -regr.intercept / regr.slope), (regr.intercept, 0.0))
                else:
                    return ((0.0, 10000.0), (regr.intercept, regr.intercept))

            if horizontal_regr and vertical_regr:
                # Use regression with better fit (higher rvalue)
                if horizontal_regr.rvalue < vertical_regr.rvalue:
                    modal_lines[c] = _vertical(vertical_regr)
                else:
                    modal_lines[c] = _horizontal(horizontal_regr)

            elif horizontal_regr:
                modal_lines[c] = _horizontal(horizontal_regr)

            elif vertical_regr:
                modal_lines[c] = _vertical(vertical_regr)

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

        # TODO(bassosimone): subs is a sympy specific feature that
        # seems quite difficult to reimplement in yakof
        new_constraints = []
        for constraint in self.constraints:
            new_constraints.append(
                Constraint(
                    constraint.usage.subs(change_indexes),
                    constraint.capacity.subs(change_capacities),
                    group=constraint.group,
                    name=constraint.name,
                )
            )

        return Model(
            new_name, self.cvs, self.pvs, new_indexes, new_capacities, new_constraints
        )
