"""
Graph Executor
==============

Executes a graph assuming nodes have been sorted in insertion
order using the `frontend.linearize` module.
"""

# TODO: ensure we document assumptions regarding the linearization

from dataclasses import dataclass, field
from typing import Callable, cast

import numpy as np

from ..frontend import graph
from . import dispatch


# TODO: explain that placeholders should be provided as values

# TODO: need to document how break is implemented and how print is
# implemented, which it makes sense that is a bit executor-dependent
# and will allow us to evolve our code


@dataclass(frozen=True)
class State:
    """
    The graph executor state.

    Attributes:
        values: A dictionary caching the result of the computation.
        flags: Bitmask containing globla node debug flags.
    """

    values: dict[graph.Node, np.ndarray]
    flags: int = 0


def __print_node(state: State, node: graph.Node):
    print("=== begin trace ===")
    print(f"name: {node.name}")
    print(f"id: {node.id}")
    print(f"type: {node.__class__}")


def __print_result(value: np.ndarray):
    print(value)
    print("=== end trace ===")
    print("")


def evaluate(state: State, node: graph.Node) -> np.ndarray:
    """
    Execute a graph by visiting nodes in order.

    Args:
        state: The current executor state.
        node: The node to evaluate.
    """

    # 1. check whether node has been already evaluated
    if node in state.values:
        return state.values[node]

    # 2. check whether we need to trace this node
    flags = state.flags | node.flags
    if flags & graph.NODE_FLAG_TRACE != 0:
        __print_node(state, node)

    # 3. evaluate the node proper
    result = _evaluate(state, node)

    # 4. check whether we need to print the computation result
    if flags & graph.NODE_FLAG_TRACE != 0:
        __print_result(result)

    # 5. check whether we need to stop after evaluating this node
    if node.name != "" and flags & graph.NODE_FLAG_BREAK != 0:
        input("executor: press any key to continue...")
        print("")

    # 6. store the node result in the state
    state.values[node] = result

    # 7. return the result
    return result


def __eval_constant_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.constant, node)
    return np.asarray(node.value)


def __eval_placeholder_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.placeholder, node)
    result = state.values.get(node)
    if result is None:
        if node.default_value is not None:
            return np.asarray(node.default_value)
        raise ValueError(f"executor: no value provided for placeholder '{node.name}'")
    return result


def __eval_binary_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.BinaryOp, node)
    left = state.values[node.left]
    right = state.values[node.right]
    try:
        return dispatch.binary_operations[type(node)](left, right)
    except KeyError:
        raise TypeError(f"executor: unknown binary operation: {type(node)}")


def __eval_unary_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.UnaryOp, node)
    operand = state.values[node.node]
    try:
        return dispatch.unary_operations[type(node)](operand)
    except KeyError:
        raise TypeError(f"executor: unknown unary operation: {type(node)}")


def __eval_where_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.where, node)
    return np.where(
        state.values[node.condition],
        state.values[node.then],
        state.values[node.otherwise],
    )


def __eval_multi_clause_where_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.multi_clause_where, node)
    conditions = []
    values = []
    for cond, value in node.clauses:
        conditions.append(state.values[cond])
        values.append(state.values[value])
    default = state.values[node.default_value]
    return np.select(conditions, values, default=default)


def __eval_axis_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.AxisOp, node)
    operand = state.values[node.node]
    try:
        return dispatch.axes_operations[type(node)](operand, node.axis)
    except KeyError:
        raise TypeError(f"executor: unknown axis operation: {type(node)}")


_evaluators: tuple[tuple[type, Callable[[State, graph.Node], np.ndarray]], ...] = (
    (graph.constant, __eval_constant_op),
    (graph.placeholder, __eval_placeholder_op),
    (graph.BinaryOp, __eval_binary_op),
    (graph.UnaryOp, __eval_unary_op),
    (graph.where, __eval_where_op),
    (graph.multi_clause_where, __eval_multi_clause_where_op),
    (graph.AxisOp, __eval_axis_op),
)


def _evaluate(state: State, node: graph.Node) -> np.ndarray:

    # Attempt to match with every possible evaluator
    for node_type, evaluator in _evaluators:
        if isinstance(node, node_type):
            return evaluator(state, node)

    # Otherwise, just bail
    raise TypeError(f"executor: unknown node type: {type(node)}")
