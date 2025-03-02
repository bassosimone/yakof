"""
NumPy Reference Evaluator
=========================

Evaluates a `graph.Node` by calling the corresponding NumPy
functions and producing a `numpy.ndarray` as output.

We consider this implementation the reference implementation and we
strive to keep it simple and easy to understand. We may write more
complex implementations if benchmarks show there are bottlenecks, but
we generally wish to keep a simple implementation around.

Design Decisions
---------------
1. State Management:
   - Uses Protocol to define state interface
   - Separates bindings from caching concerns
   - Provides both caching and non-caching implementations
   - Simple cache invalidation on input changes

2. Lazy Evaluation:
   - Optional caching of intermediate results
   - Prevents redundant computations within a single evaluation
   - Cache invalidation on placeholder value changes
   - Clear semantics for cache behavior

3. Debug Support:
   - Integrated tracepoints and breakpoints
   - Rich debug output including shape information
   - Non-intrusive to normal evaluation path
   - Bug: due to recursive evaluation, trace output is post-order (after children
     evaluation) which does not help when there is an exception when evaluating
     a specific node. We will introduce topological evaluation to get rid of this
     problem and allow for a better debugging experience.

4. Error Handling:
    - Validation of placeholder bindings
   - Descriptive error messages
   - Type checking for operations

Implementation Notes
------------------
The evaluator uses operation-type dispatch tables to keep the code
maintainable and extensible. New operations can be added by extending
the dispatch tables without modifying the core evaluation logic.

The State protocol allows for different state management strategies while
maintaining a clear contract for how state (including caching) should behave.
The reference implementation provides two state classes:
- StateWithoutCache: Simple bindings management without caching
- StateWithCache: Adds caching with full invalidation on input changes
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Protocol, runtime_checkable

import numpy as np

from ..frontend import graph, pretty
from . import debug, dispatch


@runtime_checkable
class State(Protocol):
    """Protocol representing the evaluator state.

    This protocol defines the interface for managing computation state,
    including placeholder bindings and optional caching of intermediate
    results.

    Methods:
        get_node_value: Returns cached value for a node if available
        set_node_value: Caches a computed node value
        set_placeholder_value: Updates a placeholder's value
        get_placeholder_value: Retrieves a placeholder's current value

    Note:
        Setting a placeholder value after construction invalidates at least part
        of the already-computed nodes cache and possibly the whole cache.

    Example:
        >>> state = StateWithCache({"x": np.array(1.0)})
        >>> state.set_placeholder_value("y", np.array(2.0))
        >>> state.get_placeholder_value("x")
        array(1.)
    """

    def set_node_value(self, key: graph.Node, value: np.ndarray) -> None: ...

    def get_node_value(self, key: graph.Node) -> np.ndarray | None: ...

    def set_placeholder_value(self, key: str, value: np.ndarray) -> None: ...

    def get_placeholder_value(self, key: str) -> np.ndarray | None: ...


Bindings = dict[str, np.ndarray]
"""Type alias for a dictionary of variable bindings."""


class StateWithoutCache:
    """Implementation of State protocol without result caching.

    This implementation only manages placeholder bindings without
    caching intermediate computation results. Useful when memory
    is constrained or when each node needs to be re-evaluated.
    """

    def __init__(self, bindings: Bindings) -> None:
        self._bindings = bindings

    def set_node_value(self, key: graph.Node, value: np.ndarray) -> None:
        pass

    def get_node_value(self, key: graph.Node) -> np.ndarray | None:
        return None

    def set_placeholder_value(self, key: str, value: np.ndarray) -> None:
        self._bindings[key] = value

    def get_placeholder_value(self, key: str) -> np.ndarray | None:
        if key not in self._bindings:
            return None
        return self._bindings[key]


class StateWithCache(StateWithoutCache):
    """Implementation of State protocol with result caching.

    This implementation caches intermediate computation results
    for reuse. The entire cache is invalidated when any placeholder
    value changes, ensuring computation correctness at the cost of
    potentially redundant recomputations.
    """

    def __init__(self, bindings: Bindings) -> None:
        super().__init__(bindings)
        self._cache: dict[graph.Node, np.ndarray] = {}

    def set_node_value(self, key: graph.Node, value: np.ndarray) -> None:
        self._cache[key] = value

    def get_node_value(self, key: graph.Node) -> np.ndarray | None:
        if key not in self._cache:
            return None
        return self._cache[key]

    def set_placeholder_value(self, key: str, value: np.ndarray) -> None:
        super().set_placeholder_value(key, value)
        self._cache.clear()

    def get_placeholder_value(self, key: str) -> np.ndarray | None:
        return super().get_placeholder_value(key)


def _print_node_evaluation(
    node: graph.Node, value: np.ndarray, cached: bool = False
) -> None:
    """Print node information and result after evaluation.

    This function prints comprehensive information about a node after it
    has been evaluated, including its metadata, formula, and computed result.
    """
    debug.print_graph_node(node)
    debug.print_evaluated_node(value, cached)


def evaluate(node: graph.Node, state: State) -> np.ndarray:
    """Evaluates a computation graph node to a NumPy array.

    This function performs a depth-first traversal of the computation graph,
    evaluating each node's inputs before computing its result.

    The evaluation strategy is:

    1. Check the state's cache for previously computed results
    2. For leaf nodes (constants/placeholders):
       - Convert to numpy arrays
       - Validate types and values
    3. For operation nodes:
       - Recursively evaluate inputs
       - Apply corresponding numpy operation
       - Handle broadcasting and shape compatibility
    4. Apply any debug operations (trace/break)
    5. Cache, if enabled, and return result

    Args:
        node: Root node of the computation graph to evaluate
        state: Execution state including placeholder values and cache

    Returns:
        Computed numpy array result

    Raises:
        TypeError: if we don't handle a specific node type
        ValueError: when there's no placeholder value
    """
    # Check cache first
    cached_result = state.get_node_value(node)
    if cached_result is not None:
        if node.flags & graph.NODE_FLAG_TRACE != 0:
            _print_node_evaluation(node, cached_result, cached=True)
        return cached_result

    # Compute result
    result = _evaluate(node, state)

    # Handle debug operations
    if node.flags & graph.NODE_FLAG_TRACE != 0:
        _print_node_evaluation(node, result, cached=False)
    if node.flags & graph.NODE_FLAG_BREAK != 0:
        input("Press any key to continue...")

    # Cache result
    state.set_node_value(node, result)

    return result


def _evaluate(node: graph.Node, state: State) -> np.ndarray:
    """Internal caching-agnostic implementation of the evaluate function."""

    # Constant operation
    if isinstance(node, graph.constant):
        return np.asarray(node.value)

    # Placeholder operation
    if isinstance(node, graph.placeholder):
        value = state.get_placeholder_value(node.name)
        if value is None:
            if node.default_value is not None:
                return np.asarray(node.default_value)
            raise ValueError(
                f"evaluator: no value provided for placeholder '{node.name}'"
            )
        return value

    # Binary operations
    if isinstance(node, graph.BinaryOp):
        left = evaluate(node.left, state)
        right = evaluate(node.right, state)
        try:
            return dispatch.binary_operations[type(node)](left, right)
        except KeyError:
            raise TypeError(f"evaluator: unknown binary operation: {type(node)}")

    # Unary operations
    if isinstance(node, graph.UnaryOp):
        operand = evaluate(node.node, state)
        try:
            return dispatch.unary_operations[type(node)](operand)
        except KeyError:
            raise TypeError(f"evaluator: unknown unary operation: {type(node)}")

    # Conditional operations
    if isinstance(node, graph.where):
        return np.where(
            evaluate(node.condition, state),
            evaluate(node.then, state),
            evaluate(node.otherwise, state),
        )

    if isinstance(node, graph.multi_clause_where):
        conditions = []
        values = []
        for cond, value in node.clauses:
            conditions.append(evaluate(cond, state))
            values.append(evaluate(value, state))
        default = evaluate(node.default_value, state)
        return np.select(conditions, values, default=default)

    # Axis operations
    if isinstance(node, graph.AxisOp):
        operand = evaluate(node.node, state)
        try:
            return dispatch.axes_operations[type(node)](operand, node.axis)
        except KeyError:
            raise TypeError(f"evaluator: unknown axis operation: {type(node)}")

    raise TypeError(f"evaluator: unknown node type: {type(node)}")
