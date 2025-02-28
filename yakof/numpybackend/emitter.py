"""
NumPy Linear Format Emitter
===========================

Visitor for the graph that lowers to linear format with
explicit dependency on virtual registers.

Register Allocation Strategy
----------------------------

1. Sequential Allocation:
   - Registers assigned in program order
   - Simple and predictable numbering
   - Enables future optimization passes

2. Implicit Dependencies:
   - Register numbers encode evaluation order
   - Earlier results available to later operations
   - Supports partial evaluation
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from ..frontend import graph


Register = int
"""Type alias for virtual register."""


class Operation:
    """Base class for linear NumPy representation."""

    def __hash__(self) -> int:
        return id(self)  # hashing by identity


class constant(Operation):
    """A constant scalar value in the NumPy graph.

    Args:
        value: The scalar value to store in this node.
    """

    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class placeholder(Operation):
    """Named placeholder for a value to be provided during evaluation.

    Args:
        default_value: Optional default scalar value to use for the placeholder
        if no type is provided at evaluation time.
    """

    def __init__(self, name: str, default_value: np.ndarray | None = None) -> None:
        self.name = name
        self.default_value = default_value


class BinaryOp(Operation):
    """Base class for binary operations with broadcasting.

    Args:
        left: First input node
        right: Second input node
    """

    def __init__(self, left: Register, right: Register) -> None:
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


class UnaryOp(Operation):
    """Base class for unary operations.

    Args:
        node: Input node
    """

    def __init__(self, register: Register) -> None:
        self.register = register


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


class where(Operation):
    """Selects elements from tensors based on a condition.

    Args:
        condition: Boolean tensor
        then: Values to use where condition is True
        otherwise: Values to use where condition is False
    """

    def __init__(
        self, condition: Register, then: Register, otherwise: Register
    ) -> None:
        self.condition = condition
        self.then = then
        self.otherwise = otherwise


class multi_clause_where(Operation):
    """Selects elements from tensors based on multiple conditions.

    Args:
        clauses: List of (condition, value) pairs
    """

    def __init__(self, *clauses: tuple[Register, Register]) -> None:
        self.clauses = clauses


# Axis operations


class AxisOp(Operation):
    """Base class for axis manipulation operations.

    Args:
        node: Input tensor
        axis: Axis specification
    """

    def __init__(self, register: Register, axis: graph.Axis) -> None:
        self.register = register
        self.axis = axis


class expand_dims(AxisOp):
    """Adds new axes of size 1 to a tensor's shape."""


class squeeze(AxisOp):
    """Removes axes of size 1 from a tensor's shape."""


class reduce_sum(AxisOp):
    """Computes sum of tensor elements along specified axes."""


class reduce_mean(AxisOp):
    """Computes mean of tensor elements along specified axes."""


class Program:
    def __init__(self):
        self.operations: list[Operation] = []


def emit(node: graph.Node, program: Program) -> Register:
    """Transforms the graph into a linear, register-based NumPy program."""

    def __add(op: Operation) -> Register:
        program.operations.append(op)
        return len(program.operations) - 1

    if isinstance(node, graph.constant):
        return __add(constant(np.asarray(node.value)))

    if isinstance(node, graph.placeholder):
        return __add(
            placeholder(
                node.name,
                (np.asarray(node.default_value) if node.default_value else None),
            )
        )

    # Binary operations
    if isinstance(node, graph.BinaryOp):
        left = emit(node.left, program)
        right = emit(node.right, program)

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
            return __add(ops[type(node)](left, right))
        except KeyError:
            raise TypeError(f"emitter: unknown binary operation: {type(node)}")

    # Unary operations
    if isinstance(node, graph.UnaryOp):
        register = emit(node.node, program)

        ops = {
            graph.logical_not: logical_not,
            graph.exp: exp,
            graph.log: log,
        }

        try:
            return __add(ops[type(node)](register))
        except KeyError:
            raise TypeError(f"emitter: unknown unary operation: {type(node)}")

    # Conditional operations
    if isinstance(node, graph.where):
        return __add(
            where(
                emit(node.condition, program),
                emit(node.then, program),
                emit(node.otherwise, program),
            )
        )

    if isinstance(node, graph.multi_clause_where):
        return __add(
            multi_clause_where(
                *(
                    (emit(cond, program), emit(value, program))
                    for cond, value in node.clauses
                )
            )
        )

    # Axis operations
    if isinstance(node, graph.AxisOp):
        register = emit(node.node, program)

        ops = {
            graph.expand_dims: expand_dims,
            graph.reduce_sum: reduce_sum,
            graph.reduce_mean: reduce_mean,
        }

        try:
            return __add(ops[type(node)](register, node.axis))
        except KeyError:
            raise TypeError(f"emitter: unknown axis operation: {type(node)}")

    raise TypeError(f"emitter: unknown node type: {type(node)}")
