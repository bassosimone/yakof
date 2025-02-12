"""
NumPy IR lowering
=================

Visitor for NumPy IR that lowers to linear format with
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

from . import numpyir
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


# Shape-changing operations


class reshape(Operation):
    """Reshapes a tensor into a new shape.

    Args:
        tensor: Input tensor
        shape: New shape for the tensor
    """

    def __init__(self, register: Register, shape: graph.Shape) -> None:
        self.register = register
        self.shape = shape


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


def transform(node: numpyir.Node, program: Program) -> Register:
    """Transforms the NumPy IR into a linear NumPy program."""

    def __add(op: Operation) -> Register:
        program.operations.append(op)
        return len(program.operations) - 1

    if isinstance(node, numpyir.constant):
        return __add(constant(node.value))

    if isinstance(node, numpyir.placeholder):
        return __add(
            placeholder(
                node.name,
                (node.default_value if node.default_value is not None else None),
            )
        )

    # Binary operations
    if isinstance(node, numpyir.BinaryOp):
        left = transform(node.left, program)
        right = transform(node.right, program)

        ops = {
            numpyir.add: add,
            numpyir.subtract: subtract,
            numpyir.multiply: multiply,
            numpyir.divide: divide,
            numpyir.equal: equal,
            numpyir.not_equal: not_equal,
            numpyir.less: less,
            numpyir.less_equal: less_equal,
            numpyir.greater: greater,
            numpyir.greater_equal: greater_equal,
            numpyir.logical_and: logical_and,
            numpyir.logical_or: logical_or,
            numpyir.logical_xor: logical_xor,
            numpyir.power: power,
            numpyir.maximum: maximum,
        }

        try:
            return __add(ops[type(node)](left, right))
        except KeyError:
            raise TypeError(f"numpylower: unknown binary operation: {type(node)}")

    # Unary operations
    if isinstance(node, numpyir.UnaryOp):
        register = transform(node.node, program)

        ops = {
            numpyir.logical_not: logical_not,
            numpyir.exp: exp,
            numpyir.log: log,
        }

        try:
            return __add(ops[type(node)](register))
        except KeyError:
            raise TypeError(f"numpylower: unknown unary operation: {type(node)}")

    # Conditional operations
    if isinstance(node, numpyir.where):
        return __add(
            where(
                transform(node.condition, program),
                transform(node.then, program),
                transform(node.otherwise, program),
            )
        )

    if isinstance(node, numpyir.multi_clause_where):
        return __add(
            multi_clause_where(
                *(
                    (transform(cond, program), transform(value, program))
                    for cond, value in node.clauses
                )
            )
        )

    # Shape operations
    if isinstance(node, numpyir.reshape):
        return __add(reshape(transform(node.node, program), node.shape))

    # Axis operations
    if isinstance(node, numpyir.AxisOp):
        register = transform(node.node, program)

        ops = {
            numpyir.expand_dims: expand_dims,
            numpyir.reduce_sum: reduce_sum,
            numpyir.reduce_mean: reduce_mean,
        }

        try:
            return __add(ops[type(node)](register, node.axis))
        except KeyError:
            raise TypeError(f"numpylower: unknown axis operation: {type(node)}")

    raise TypeError(f"numpylower: unknown node type: {type(node)}")
