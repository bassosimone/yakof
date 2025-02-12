"""
NumPy evaluator
===============

Evaluates a `numpyeval.Node` by calling the corresponding NumPy
functions and producing a `numpy.ndarray` as output.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from . import numpyir
from ..frontend import graph


Bindings = dict[str, np.ndarray]
"""Type alias for a dictionary of variable bindings."""


def evaluate(node: numpyir.Node, bindings: Bindings) -> np.ndarray:
    """Evaluates a `numpyir.Node` to an `numpy.ndarray`."""

    if isinstance(node, numpyir.constant):
        return node.value

    if isinstance(node, numpyir.placeholder):
        if node.name not in bindings:
            if node.default_value is not None:
                return np.asarray(node.default_value)
            raise ValueError(
                f"numpyeval: no value provided for placeholder '{node.name}'"
            )
        return bindings[node.name]

    # Binary operations
    if isinstance(node, numpyir.BinaryOp):
        left = evaluate(node.left, bindings)
        right = evaluate(node.right, bindings)

        ops = {
            numpyir.add: np.add,
            numpyir.subtract: np.subtract,
            numpyir.multiply: np.multiply,
            numpyir.divide: np.divide,
            numpyir.equal: np.equal,
            numpyir.not_equal: np.not_equal,
            numpyir.less: np.less,
            numpyir.less_equal: np.less_equal,
            numpyir.greater: np.greater,
            numpyir.greater_equal: np.greater_equal,
            numpyir.logical_and: np.logical_and,
            numpyir.logical_or: np.logical_or,
            numpyir.logical_xor: np.logical_xor,
            numpyir.power: np.power,
            numpyir.maximum: np.maximum,
        }

        try:
            return ops[type(node)](left, right)
        except KeyError:
            raise TypeError(f"numpyeval: unknown binary operation: {type(node)}")

    # Unary operations
    if isinstance(node, numpyir.UnaryOp):
        operand = evaluate(node.node, bindings)

        ops = {
            numpyir.logical_not: np.logical_not,
            numpyir.exp: np.exp,
            numpyir.log: np.log,
        }

        try:
            return ops[type(node)](operand)
        except KeyError:
            raise TypeError(f"numpyir: unknown unary operation: {type(node)}")

    # Conditional operations
    if isinstance(node, numpyir.where):
        return np.where(
            evaluate(node.condition, bindings),
            evaluate(node.then, bindings),
            evaluate(node.otherwise, bindings),
        )

    if isinstance(node, numpyir.multi_clause_where):
        conditions = []
        values = []
        for cond, value in node.clauses[:-1]:
            conditions.append(evaluate(cond, bindings))
            values.append(evaluate(value, bindings))
        default = evaluate(node.clauses[-1][1], bindings)
        return np.select(conditions, values, default=default)

    # Shape operations
    if isinstance(node, numpyir.reshape):
        return evaluate(node.node, bindings).reshape(node.shape)

    # Axis operations
    if isinstance(node, numpyir.AxisOp):
        operand = evaluate(node.node, bindings)

        ops = {
            numpyir.expand_dims: lambda x: np.expand_dims(x, node.axis),
            numpyir.reduce_sum: lambda x: np.sum(x, axis=node.axis),
            numpyir.reduce_mean: lambda x: np.mean(x, axis=node.axis),
        }

        try:
            return ops[type(node)](operand)
        except KeyError:
            raise TypeError(f"numpyir: unknown axis operation: {type(node)}")

    raise TypeError(f"numpyir: unknown node type: {type(node)}")
