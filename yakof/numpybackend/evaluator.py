"""
NumPy evaluator
===============

Evaluates a `hir.Node` by calling the corresponding NumPy
functions and producing a `numpy.ndarray` as output.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from . import hir
from ..frontend import graph


Bindings = dict[str, np.ndarray]
"""Type alias for a dictionary of variable bindings."""


def evaluate(node: hir.Node, bindings: Bindings) -> np.ndarray:
    """Evaluates a `hir.Node` to an `numpy.ndarray`."""

    if isinstance(node, hir.constant):
        return node.value

    if isinstance(node, hir.placeholder):
        if node.name not in bindings:
            if node.default_value is not None:
                return np.asarray(node.default_value)
            raise ValueError(
                f"evaluator: no value provided for placeholder '{node.name}'"
            )
        return bindings[node.name]

    # Binary operations
    if isinstance(node, hir.BinaryOp):
        left = evaluate(node.left, bindings)
        right = evaluate(node.right, bindings)

        ops = {
            hir.add: np.add,
            hir.subtract: np.subtract,
            hir.multiply: np.multiply,
            hir.divide: np.divide,
            hir.equal: np.equal,
            hir.not_equal: np.not_equal,
            hir.less: np.less,
            hir.less_equal: np.less_equal,
            hir.greater: np.greater,
            hir.greater_equal: np.greater_equal,
            hir.logical_and: np.logical_and,
            hir.logical_or: np.logical_or,
            hir.logical_xor: np.logical_xor,
            hir.power: np.power,
            hir.maximum: np.maximum,
        }

        try:
            return ops[type(node)](left, right)
        except KeyError:
            raise TypeError(f"evaluator: unknown binary operation: {type(node)}")

    # Unary operations
    if isinstance(node, hir.UnaryOp):
        operand = evaluate(node.node, bindings)

        ops = {
            hir.logical_not: np.logical_not,
            hir.exp: np.exp,
            hir.log: np.log,
        }

        try:
            return ops[type(node)](operand)
        except KeyError:
            raise TypeError(f"evaluator: unknown unary operation: {type(node)}")

    # Conditional operations
    if isinstance(node, hir.where):
        return np.where(
            evaluate(node.condition, bindings),
            evaluate(node.then, bindings),
            evaluate(node.otherwise, bindings),
        )

    if isinstance(node, hir.multi_clause_where):
        conditions = []
        values = []
        for cond, value in node.clauses[:-1]:
            conditions.append(evaluate(cond, bindings))
            values.append(evaluate(value, bindings))
        default = evaluate(node.clauses[-1][1], bindings)
        return np.select(conditions, values, default=default)

    # Axis operations
    if isinstance(node, hir.AxisOp):
        operand = evaluate(node.node, bindings)

        ops = {
            hir.expand_dims: lambda x: np.expand_dims(x, node.axis),
            hir.reduce_sum: lambda x: np.sum(x, axis=node.axis),
            hir.reduce_mean: lambda x: np.mean(x, axis=node.axis),
        }

        try:
            return ops[type(node)](operand)
        except KeyError:
            raise TypeError(f"evaluator: unknown axis operation: {type(node)}")

    raise TypeError(f"evaluator: unknown node type: {type(node)}")
