"""
NumPy evaluator
===============

Evaluates a `graph.Node` by calling the corresponding NumPy
functions and producing a `numpy.ndarray` as output.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from ..frontend import graph, pretty


Bindings = dict[str, np.ndarray]
"""Type alias for a dictionary of variable bindings."""


Cache = dict[graph.Node, np.ndarray]
"""Type alias for a dictionary tracking already-computed node values."""


def _print_tracepoint(node: graph.Node, value: np.ndarray) -> None:
    print("=== begin tracepoint ===")
    print(f"name: {node.name}")
    print(f"formula: {pretty.format(node)}")
    print(f"shape: {value.shape}")
    print(f"value:\n{value}")
    print("=== end tracepoint ===")
    print("")


def evaluate(
    node: graph.Node,
    bindings: Bindings,
    cache: Cache | None = None,
) -> np.ndarray:
    """Evaluates a `graph.Node` to an `numpy.ndarray`."""

    # Use cache if available
    if cache and node in cache:
        return cache[node]

    # Code to run before returning
    def __before_return(value: np.ndarray) -> np.ndarray:
        if node.flags & graph.NODE_FLAG_TRACEPOINT != 0:
            _print_tracepoint(node, value)
        if node.flags & graph.NODE_FLAG_BREAKPOINT != 0:
            input("Press any key to continue...")
        if cache:
            cache[node] = value
        return value

    # Constant operation
    if isinstance(node, graph.constant):
        return __before_return(np.asarray(node.value))

    # Placeholder operation
    if isinstance(node, graph.placeholder):
        if node.name not in bindings:
            if node.default_value is not None:
                return __before_return(np.asarray(node.default_value))
            raise ValueError(
                f"evaluator: no value provided for placeholder '{node.name}'"
            )
        return __before_return(bindings[node.name])

    # Binary operations
    if isinstance(node, graph.BinaryOp):
        left = evaluate(node.left, bindings, cache)
        right = evaluate(node.right, bindings, cache)

        ops = {
            graph.add: np.add,
            graph.subtract: np.subtract,
            graph.multiply: np.multiply,
            graph.divide: np.divide,
            graph.equal: np.equal,
            graph.not_equal: np.not_equal,
            graph.less: np.less,
            graph.less_equal: np.less_equal,
            graph.greater: np.greater,
            graph.greater_equal: np.greater_equal,
            graph.logical_and: np.logical_and,
            graph.logical_or: np.logical_or,
            graph.logical_xor: np.logical_xor,
            graph.power: np.power,
            graph.maximum: np.maximum,
        }

        try:
            return __before_return(ops[type(node)](left, right))
        except KeyError:
            raise TypeError(f"evaluator: unknown binary operation: {type(node)}")

    # Unary operations
    if isinstance(node, graph.UnaryOp):
        operand = evaluate(node.node, bindings, cache)

        ops = {
            graph.logical_not: np.logical_not,
            graph.exp: np.exp,
            graph.log: np.log,
        }

        try:
            return __before_return(ops[type(node)](operand))
        except KeyError:
            raise TypeError(f"evaluator: unknown unary operation: {type(node)}")

    # Conditional operations
    if isinstance(node, graph.where):
        return __before_return(
            np.where(
                evaluate(node.condition, bindings, cache),
                evaluate(node.then, bindings, cache),
                evaluate(node.otherwise, bindings, cache),
            )
        )

    if isinstance(node, graph.multi_clause_where):
        conditions = []
        values = []
        for cond, value in node.clauses[:-1]:
            conditions.append(evaluate(cond, bindings, cache))
            values.append(evaluate(value, bindings, cache))
        default = evaluate(node.default_value, bindings, cache)
        return __before_return(np.select(conditions, values, default=default))

    # Axis operations
    if isinstance(node, graph.AxisOp):
        operand = evaluate(node.node, bindings, cache)

        ops = {
            graph.expand_dims: lambda x: np.expand_dims(x, node.axis),
            graph.reduce_sum: lambda x: np.sum(x, axis=node.axis),
            graph.reduce_mean: lambda x: np.mean(x, axis=node.axis),
        }

        try:
            return __before_return(ops[type(node)](operand))
        except KeyError:
            raise TypeError(f"evaluator: unknown axis operation: {type(node)}")

    raise TypeError(f"evaluator: unknown node type: {type(node)}")
