"""
NumPy evaluator
===============

Evaluates a `graph.Node` by calling the corresponding NumPy
functions and producing a `numpy.ndarray` as output.

We consider this implementation the reference implementation and we
strive to keep it simple and easy to understand. We may write more
complex implementations if benchmarks show there are bottlenecks, but
we generally strive to keep a simple implementation around.

Design Decisions
---------------
1. Lazy Evaluation:
   - Uses dictionary-based caching
   - Prevents redundant computations
   - Preserves DAG semantics

2. Debug Support:
   - Integrated tracepoints and breakpoints
   - Rich debug output including shape information
   - Non-intrusive to normal evaluation path

3. Error Handling:
   - Descriptive error messages
   - Type checking for operations
   - Validation of placeholder bindings

Implementation Notes
------------------
The evaluator uses operation-type dispatch tables to keep the code
maintainable and extensible. New operations can be added by extending
the dispatch tables without modifying the core evaluation logic.
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
    """Evaluates a computation graph node to a NumPy array.

    This function performs a depth-first traversal of the computation graph,
    evaluating each node's inputs before computing its result.

    The evaluation strategy is:

    1. Check cache for previously computed results
    2. For leaf nodes (constants/placeholders):
       - Convert to numpy arrays
       - Validate types and values
    3. For operation nodes:
       - Recursively evaluate inputs
       - Apply corresponding numpy operation
       - Handle broadcasting and shape compatibility
    4. Apply any debug operations (trace/break)
    5. Cache and return result

    Args:
        node: Root node of the computation graph to evaluate
        bindings: Dictionary mapping placeholder names to concrete values
        cache: Optional cache of previously computed node values

    Returns:
        Computed numpy array result

    Raises:
        TypeError: if we don't handle a specific node type
        ValueError: when there's no placeholder value
    """

    # Use cache if available
    if cache and node in cache:
        return cache[node]

    # Code to run before returning
    def __before_return(value: np.ndarray) -> np.ndarray:
        if node.flags & graph.NODE_FLAG_TRACE != 0:
            _print_tracepoint(node, value)
        if node.flags & graph.NODE_FLAG_BREAK != 0:
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
