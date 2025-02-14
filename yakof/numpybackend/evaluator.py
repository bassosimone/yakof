"""
NumPy evaluator
===============

Evaluates a `semtree.Node` by calling the corresponding NumPy
functions and producing a `numpy.ndarray` as output.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from . import semtree
from ..frontend import graph


Bindings = dict[str, np.ndarray]
"""Type alias for a dictionary of variable bindings."""


def evaluate(node: semtree.Node, bindings: Bindings) -> np.ndarray:
    """Evaluates a `semtree.Node` to an `numpy.ndarray`."""

    if isinstance(node, semtree.constant):
        return node.value

    if isinstance(node, semtree.placeholder):
        if node.name not in bindings:
            if node.default_value is not None:
                return np.asarray(node.default_value)
            raise ValueError(
                f"evaluator: no value provided for placeholder '{node.name}'"
            )
        return bindings[node.name]

    # Binary operations
    if isinstance(node, semtree.BinaryOp):
        left = evaluate(node.left, bindings)
        right = evaluate(node.right, bindings)

        ops = {
            semtree.add: np.add,
            semtree.subtract: np.subtract,
            semtree.multiply: np.multiply,
            semtree.divide: np.divide,
            semtree.equal: np.equal,
            semtree.not_equal: np.not_equal,
            semtree.less: np.less,
            semtree.less_equal: np.less_equal,
            semtree.greater: np.greater,
            semtree.greater_equal: np.greater_equal,
            semtree.logical_and: np.logical_and,
            semtree.logical_or: np.logical_or,
            semtree.logical_xor: np.logical_xor,
            semtree.power: np.power,
            semtree.maximum: np.maximum,
        }

        try:
            return ops[type(node)](left, right)
        except KeyError:
            raise TypeError(f"evaluator: unknown binary operation: {type(node)}")

    # Unary operations
    if isinstance(node, semtree.UnaryOp):
        operand = evaluate(node.node, bindings)

        ops = {
            semtree.logical_not: np.logical_not,
            semtree.exp: np.exp,
            semtree.log: np.log,
        }

        try:
            return ops[type(node)](operand)
        except KeyError:
            raise TypeError(f"evaluator: unknown unary operation: {type(node)}")

    # Conditional operations
    if isinstance(node, semtree.where):
        return np.where(
            evaluate(node.condition, bindings),
            evaluate(node.then, bindings),
            evaluate(node.otherwise, bindings),
        )

    if isinstance(node, semtree.multi_clause_where):
        conditions = []
        values = []
        for cond, value in node.clauses[:-1]:
            conditions.append(evaluate(cond, bindings))
            values.append(evaluate(value, bindings))
        default = evaluate(node.clauses[-1][1], bindings)
        return np.select(conditions, values, default=default)

    # Axis operations
    if isinstance(node, semtree.AxisOp):
        operand = evaluate(node.node, bindings)

        ops = {
            semtree.expand_dims: lambda x: np.expand_dims(x, node.axis),
            semtree.reduce_sum: lambda x: np.sum(x, axis=node.axis),
            semtree.reduce_mean: lambda x: np.mean(x, axis=node.axis),
        }

        try:
            return ops[type(node)](operand)
        except KeyError:
            raise TypeError(f"evaluator: unknown axis operation: {type(node)}")

    raise TypeError(f"evaluator: unknown node type: {type(node)}")
