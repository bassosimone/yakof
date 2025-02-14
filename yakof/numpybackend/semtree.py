"""
NumPy Semantic Tree
===================

The semtree serves several purposes:

1. Backend Specificity:
   - Represents operations in terms of NumPy primitives
   - Enables NumPy-specific optimizations
   - Provides clean separation from frontend abstractions

2. Type Safety:
   - Maintains strong typing through transformation
   - Ensures frontend guarantees carry through to backend

3. Optimization Opportunity:
   - Enables backend-specific optimizations like:
     * Operation fusion
     * Common subexpression elimination
     * Dead code elimination

Why a Separate SemTree?
-----------------------

While the frontend graph could be evaluated directly, having a
NumPy-specific semtree enables:

- Cleaner separation of concerns
- Potential backend-specific optimizations
- Seamless support of multiple backends
- More streamlined lowering to linear form
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from ..frontend import graph


class Node:
    """Base class for all NumPy graph nodes."""

    def __hash__(self) -> int:
        return id(self)  # hashing by identity


class constant(Node):
    """A constant scalar value in the NumPy graph.

    Args:
        value: The scalar value to store in this node.
    """

    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class placeholder(Node):
    """Named placeholder for a value to be provided during evaluation.

    Args:
        default_value: Optional default scalar value to use for the placeholder
        if no type is provided at evaluation time.
    """

    def __init__(self, name: str, default_value: np.ndarray | None = None) -> None:
        self.name = name
        self.default_value = default_value


class BinaryOp(Node):
    """Base class for binary operations with broadcasting.

    Args:
        left: First input node
        right: Second input node
    """

    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right


# Arithmetic operations


class add(BinaryOp):
    """Element-wise addition of two tensors."""


class subtract(BinaryOp):
    """Element-wise subtraction of two tensors."""


class multiply(BinaryOp):
    """Element-wise multiplication of two tensors."""


class divide(BinaryOp):
    """Element-wise division of two tensors."""


# Comparison operations


class equal(BinaryOp):
    """Element-wise equality comparison of two tensors."""


class not_equal(BinaryOp):
    """Element-wise inequality comparison of two tensors."""


class less(BinaryOp):
    """Element-wise less-than comparison of two tensors."""


class less_equal(BinaryOp):
    """Element-wise less-than-or-equal comparison of two tensors."""


class greater(BinaryOp):
    """Element-wise greater-than comparison of two tensors."""


class greater_equal(BinaryOp):
    """Element-wise greater-than-or-equal comparison of two tensors."""


# Logical operations


class logical_and(BinaryOp):
    """Element-wise logical AND of two boolean tensors."""


class logical_or(BinaryOp):
    """Element-wise logical OR of two boolean tensors."""


class logical_xor(BinaryOp):
    """Element-wise logical XOR of two boolean tensors."""


class UnaryOp(Node):
    """Base class for unary operations.

    Args:
        node: Input node
    """

    def __init__(self, node: Node) -> None:
        self.node = node


class logical_not(UnaryOp):
    """Element-wise logical NOT of a boolean tensor."""


# Math operations


class exp(UnaryOp):
    """Element-wise exponential of a tensor."""


class power(BinaryOp):
    """Element-wise power operation (first tensor raised to power of second)."""


pow = power
"""Alias for power for comaptibility with NumPy naming."""


class log(UnaryOp):
    """Element-wise natural logarithm of a tensor."""


class maximum(BinaryOp):
    """Element-wise maximum of two tensors."""


# Conditional operations


class where(Node):
    """Selects elements from tensors based on a condition.

    Args:
        condition: Boolean tensor
        then: Values to use where condition is True
        otherwise: Values to use where condition is False
    """

    def __init__(self, condition: Node, then: Node, otherwise: Node) -> None:
        self.condition = condition
        self.then = then
        self.otherwise = otherwise


class multi_clause_where(Node):
    """Selects elements from tensors based on multiple conditions.

    Args:
        clauses: List of (condition, value) pairs
    """

    def __init__(self, *clauses: tuple[Node, Node]) -> None:
        self.clauses = clauses


# Shape-changing operations


class AxisOp(Node):
    """Base class for axis manipulation operations.

    Args:
        node: Input tensor
        axis: Axis specification
    """

    def __init__(self, node: Node, axis: graph.Axis) -> None:
        self.node = node
        self.axis = axis


class expand_dims(AxisOp):
    """Adds new axes of size 1 to a tensor's shape."""


class squeeze(AxisOp):
    """Removes axes of size 1 from a tensor's shape."""


class reduce_sum(AxisOp):
    """Computes sum of tensor elements along specified axes."""


class reduce_mean(AxisOp):
    """Computes mean of tensor elements along specified axes."""


def transform(gr_node: graph.Node) -> Node:
    """Transforms the abstract graph into a NumPy-specific graph."""

    if isinstance(gr_node, graph.constant):
        return constant(np.asarray(gr_node.value))

    if isinstance(gr_node, graph.placeholder):
        return placeholder(
            gr_node.name,
            (
                np.asarray(gr_node.default_value)
                if gr_node.default_value is not None
                else None
            ),
        )

    # Binary operations
    if isinstance(gr_node, graph.BinaryOp):
        left = transform(gr_node.left)
        right = transform(gr_node.right)

        ops = {
            graph.add: add,
            graph.subtract: subtract,
            graph.multiply: multiply,
            graph.divide: divide,
            graph.equal: equal,
            graph.not_equal: not_equal,
            graph.less: less,
            graph.less_equal: less_equal,
            graph.greater: greater,
            graph.greater_equal: greater_equal,
            graph.logical_and: logical_and,
            graph.logical_or: logical_or,
            graph.logical_xor: logical_xor,
            graph.power: power,
            graph.maximum: maximum,
        }

        try:
            return ops[type(gr_node)](left, right)
        except KeyError:
            raise TypeError(f"semtree: unknown binary operation: {type(gr_node)}")

    # Unary operations
    if isinstance(gr_node, graph.UnaryOp):
        np_node = transform(gr_node.node)

        ops = {
            graph.logical_not: logical_not,
            graph.exp: exp,
            graph.log: log,
        }

        try:
            return ops[type(gr_node)](np_node)
        except KeyError:
            raise TypeError(f"semtree: unknown unary operation: {type(gr_node)}")

    # Conditional operations
    if isinstance(gr_node, graph.where):
        return where(
            transform(gr_node.condition),
            transform(gr_node.then),
            transform(gr_node.otherwise),
        )

    if isinstance(gr_node, graph.multi_clause_where):
        return multi_clause_where(
            *((transform(cond), transform(value)) for cond, value in gr_node.clauses)
        )

    # Axis operations
    if isinstance(gr_node, graph.AxisOp):
        np_node = transform(gr_node.node)

        ops = {
            graph.expand_dims: expand_dims,
            graph.reduce_sum: reduce_sum,
            graph.reduce_mean: reduce_mean,
        }

        try:
            return ops[type(gr_node)](np_node, gr_node.axis)
        except KeyError:
            raise TypeError(f"semtree: unknown axis operation: {type(gr_node)}")

    raise TypeError(f"semtree: unknown node type: {type(gr_node)}")
