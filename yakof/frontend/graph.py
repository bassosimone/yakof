"""
Computation Graph Building
==========================

This module allows to build an abstract computation graph using TensorFlow-like
computation primitives and concepts. These primitives and concepts are similar to
NumPy primitives, but we picked up TensorFlow ones when they disagree.

This module provides:

1. Basic node types for constants and placeholders
2. Arithmetic operations (add, subtract, multiply, divide)
3. Comparison operations (equal, not_equal, less, less_equal, greater, greater_equal)
4. Logical operations (and, or, xor, not)
5. Mathematical operations (exp, power, log)
6. Shape manipulation operations (reshape, expand_dims, squeeze)
7. Reduction operations (sum, mean)

The nodes form a directed acyclic graph (DAG) that represents computations
to be performed. Each node implements a specific operation and stores its
inputs as attributes. The graph can then be evaluated by traversing the nodes
and performing their operations using NumPy, TensorFlow, etc.

Here's an example of what you can do with this module:

    >>> from yakof.frontend import graph
    >>> a = graph.placeholder("a", 1.0)
    >>> b = graph.constant(2.0)
    >>> c = graph.add(a, b)
    >>> d = grap.multiply(c, c)

Like TensorFlow, we support placeholders. That is, variables with a given
name that can be filled in at execution time with concrete values. We also
support constants, which must be bool, float, or int scalars.

Because our goal is to *capture* the arguments provided to function invocations
for later evaluation, we are using classes instead of functions. (We could
alternatively have used closures, but it would have been more clumsy.) To keep
the invoked entities names as close as possible to TensorFlow, we named the
classes using snake_case rather than CmaleCase. This is a pragmatic and conscious
choice: violating PEP8 to produce more readable code.

The main type in this module is the `Node`, representing a node in the
computation graph. Each operation (e.g., `add`) is a subclass of the `Node`
capturing the arguments it has been provided on construction.

Design Decisions
----------------

1. Class-based vs Function-based:
   - Classes capture operation arguments naturally
   - Enable visitor pattern for transformations
   - Allow future addition of operation-specific attributes

2. Snake Case Operation Names:
   - Match NumPy/TensorFlow conventions
   - Improve readability in mathematical context
   - Enable direct mapping to backend operations

3. Node Identity:
   - Nodes are identified by their instance identity
   - Enables graph traversal and transformation
   - Supports future optimization passes
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


Axis = int | tuple[int, ...]
"""Type alias for axis specifications in shape operations."""

Scalar = bool | float | int
"""Type alias for supported scalar value types."""

Shape = tuple[int, ...]
"""Type alias for tensor shape specifications."""


class Node:
    """Base class for all computation graph nodes."""

    def __hash__(self) -> int:
        return id(self)  # hashing by identity


class constant(Node):
    """A constant scalar value in the computation graph.

    Args:
        value: The scalar value to store in this node.
    """

    def __init__(self, value: Scalar) -> None:
        self.value = value


class placeholder(Node):
    """Named placeholder for a value to be provided during evaluation.

    Args:
        default_value: Optional default scalar value to use for the placeholder
        if no type is provided at evaluation time.
    """

    def __init__(self, name: str, default_value: Scalar | None = None) -> None:
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


class reshape(Node):
    """Reshapes a tensor into a new shape.

    Args:
        tensor: Input tensor
        shape: New shape for the tensor
    """

    def __init__(self, node: Node, shape: Shape) -> None:
        self.node = node
        self.shape = shape


class AxisOp(Node):
    """Base class for axis manipulation operations.

    Args:
        node: Input tensor
        axis: Axis specification
    """

    def __init__(self, node: Node, axis: Axis) -> None:
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
