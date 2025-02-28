"""
Abstract Tensor Operations
==========================

This modules defines tensors spaces and tensors within spaces. It builds on
top of the computation graph (`graph.py`) to provide a type-safe interface for
working with tensors in different tensor spaces.

The module provides these abstractions:

1. TensorSpace: A space of tensors with a given basis. Provides
mathematical operations such as exp, log, and power.

2. Tensor: A tensor with associated basis vectors. Supports
arithmetic, comparison, and logical operations.

The type parameters ensure that operations between tensors are only
allowed when they share the same basis of the space.

On mathematical terminology
---------------------------

This package uses 'tensor' in the computational sense (i.e., multidimensional
arrays) and borrows mathematical concepts such as bases and vector spaces to
use a terminology that would sound familiar to engineers.

Categorical Structure
---------------------

The module implements a categorical structure where:

1. Objects are tensor spaces (TensorSpace)
2. Morphisms are structure-preserving maps between tensor spaces

This module only implements endomorphisms, i.e., morphisms from a tensor space
to itself. Morphisms between different tensor spaces are implemented in the
`morphisms.py` module.

This categorical view informs key design decisions:

1. Operations that change tensor shape/basis (like reduce_sum or expand_dims)
are not methods of TensorSpace as they are morphisms between different spaces
rather than endomorphisms within a space (see `morphisms.py`).

2. Operations that preserve tensor structure (like add or multiply) are methods
of TensorSpace as they are endomorphisms within the same space.

3. The type system enforces these categorical constraints at compile time,
ensuring a certain degree of mathematical correctness.

Type System Implementation
--------------------------

The type system uses generics to enforce:

1. Basis compatibility:
- Operations allowed only between tensors with the same basis
- Compile-time detection of bases mismatches

2. Context preservation:
- Clear distinction between different tensor spaces
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Generic, Protocol, Sequence, TypeVar, runtime_checkable

from . import graph


@runtime_checkable
class Basis(Protocol):
    """Protocol defining the interface for tensor space bases.

    All bases must provide their axes as an ordered tuple of integers, establishing
    their position in the canonical ordering. To generate the canonical
    ordering, use the morphisms.generate_canonical_axes function.

    Examples:
        >>> class XYBasis:
        ...     axes = (1000, 1001)  # X and Y axes
        >>> isinstance(XYBasis(), Basis)
        True
    """

    axes: graph.Axis


def ensure_same_basis(left: Any, right: Any) -> None:
    """
    Ensures that the two bases are the same.

    The two bases are the same if

    1. they are the same instance (tested using the `is` operator); or

    2. they both implement Basis and return the same axes.

    If these two conditions are not met, this function raise a TypeError. Note that
    using a type checker prevents this class of errors at compile time.
    """
    if left is not right:
        if not isinstance(left, Basis):
            raise TypeError(f"{left} must be an instance of Basis")
        if not isinstance(right, Basis):
            raise TypeError(f"{right} must be an instance of Basis")
        if left.axes != right.axes:
            raise ValueError("Tensors must have the same basis")


B = TypeVar("B")
"""Type variable for tensor basis types, used by Tensor and TensorSpace."""

C = TypeVar("C")
"""Type variable for condition tensor basis types, used by where and multi_clause_where."""


class Tensor(Generic[B]):
    """A tensor with associated basis vectors.

    Type Parameters:
        B: Type of the basis vectors.

    Attributes:
        space: The tensor space of the tensor.
        node: The computation graph node representing the tensor.

    Args:
        space: The tensor space of the tensor.
        node: The computation graph node representing the tensor.

    Implementation Note
    -------------------

    We use ensure_same_basis whenever we combine tensors potentially
    belonging to different tensor spaces.
    """

    def __init__(self, space: TensorSpace[B], node: graph.Node) -> None:
        self.space = space
        self.node = node

    # autonaming.Namer protocol implementation
    def implements_namer(self) -> None:
        """This method is part of the autonaming.Namer protocol"""

    @property
    def name(self) -> str:
        """This method is part of the autonaming.Namer protocol"""
        return self.node.name

    @name.setter
    def name(self, value: str) -> None:
        """This method is part of the autonaming.Namer protocol"""
        self.node.name = value

    # Hashing by identity - see also comment in graph.py
    def __hash__(self) -> int:
        return hash(self.node)

    # Arithmetic operators
    def __add__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.add(self, self.space.ensure_tensor(other))

    def __radd__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.add(self.space.ensure_tensor(other), self)

    def __sub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.subtract(self, self.space.ensure_tensor(other))

    def __rsub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.subtract(self.space.ensure_tensor(other), self)

    def __mul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.multiply(self, self.space.ensure_tensor(other))

    def __rmul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.multiply(self.space.ensure_tensor(other), self)

    def __truediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.divide(self, self.space.ensure_tensor(other))

    def __rtruediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.divide(self.space.ensure_tensor(other), self)

    # Comparison operators
    #
    # See corresponding comment in graph.py
    def __eq__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return self.space.equal(self, self.space.ensure_tensor(other))

    def __ne__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return self.space.not_equal(self, self.space.ensure_tensor(other))

    def __lt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.less(self, self.space.ensure_tensor(other))

    def __le__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.less_equal(self, self.space.ensure_tensor(other))

    def __gt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.greater(self, self.space.ensure_tensor(other))

    def __ge__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.greater_equal(self, self.space.ensure_tensor(other))

    # Logical operators
    def __and__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.logical_and(self, self.space.ensure_tensor(other))

    def __rand__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.logical_and(self.space.ensure_tensor(other), self)

    def __or__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.logical_or(self, self.space.ensure_tensor(other))

    def __ror__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.logical_or(self.space.ensure_tensor(other), self)

    def __xor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.logical_xor(self, self.space.ensure_tensor(other))

    def __rxor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return self.space.logical_xor(self.space.ensure_tensor(other), self)

    def __invert__(self) -> Tensor[B]:
        return self.space.logical_not(self)


class TensorSpace(Generic[B]):
    """A space of tensors with associated basis vectors.

    Type Parameters:
        B: Type of the basis vectors.

    Attributes:
        basis: The basis of the tensor space.

    Args:
        basis: The basis of the tensor space.

    Implementation Note
    -------------------

    We use ensure_same_basis whenever we combine tensors potentially
    belonging to different tensor spaces.
    """

    def __init__(self, basis: B) -> None:
        self.basis = basis

    def axes(self) -> graph.Axis:
        if not isinstance(self.basis, Basis):
            raise TypeError(f"{self.basis} must be an instance of Basis")
        return self.basis.axes

    def new_tensor(self, node: graph.Node) -> Tensor[B]:
        """Creates a new tensor in this space given a graph node."""
        return Tensor[B](self, node)

    def placeholder(
        self,
        name: str,
        default_value: graph.Scalar | None = None,
    ) -> Tensor[B]:
        """Creates a placeholder tensor."""
        return self.new_tensor(graph.placeholder(name, default_value))

    def constant(self, value: graph.Scalar, name: str = "") -> Tensor[B]:
        """Creates a constant tensor."""
        return self.new_tensor(graph.constant(value, name))

    def ensure_tensor(self, t: Tensor[B] | graph.Scalar) -> Tensor[B]:
        """Converts a scalar to a tensor if needed."""
        if isinstance(t, graph.Scalar):
            t = self.constant(t)
        return t

    def exp(self, t: Tensor[B]) -> Tensor[B]:
        """Compute the exponential of a tensor."""
        ensure_same_basis(self.basis, t.space.basis)
        return self.new_tensor(graph.exp(t.node))

    def power(self, t: Tensor[B], exp: Tensor[B]) -> Tensor[B]:
        """Raise tensor to the power of another tensor."""
        ensure_same_basis(self.basis, t.space.basis)
        ensure_same_basis(self.basis, exp.space.basis)
        return self.new_tensor(graph.power(t.node, exp.node))

    def log(self, t: Tensor[B]) -> Tensor[B]:
        """Compute the natural logarithm of a tensor."""
        ensure_same_basis(self.basis, t.space.basis)
        return self.new_tensor(graph.log(t.node))

    def maximum(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Compute element-wise maximum of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.maximum(t1.node, t2.node))

    def where(
        self, cond: Tensor[C], then: Tensor[B], otherwise: Tensor[B]
    ) -> Tensor[B]:
        """Select elements based on condition."""
        ensure_same_basis(self.basis, then.space.basis)
        ensure_same_basis(self.basis, otherwise.space.basis)
        return self.new_tensor(graph.where(cond.node, then.node, otherwise.node))

    def multi_clause_where(
        self,
        clauses: Sequence[tuple[Tensor[C], Tensor[B]]],
        default_value: Tensor[B] | graph.Scalar,
    ) -> Tensor[B]:
        """Select elements based on multiple conditions."""
        for cond, value in clauses:
            ensure_same_basis(self.basis, value.space.basis)

        default_value = self.ensure_tensor(default_value)
        ensure_same_basis(self.basis, default_value.space.basis)

        return self.new_tensor(
            graph.multi_clause_where(
                clauses=[(cond.node, value.node) for cond, value in clauses],
                default_value=default_value.node,
            ),
        )

    def tracepoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a tracepoint for the current tensor inside the computation graph."""
        ensure_same_basis(self.basis, t.space.basis)
        return self.new_tensor(graph.tracepoint(t.node))

    def breakpoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a breakpoint for the current tensor inside the computation graph."""
        ensure_same_basis(self.basis, t.space.basis)
        return self.new_tensor(graph.breakpoint(t.node))

    # Additional shape/structure preserving operations
    def add(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise addition of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.add(t1.node, t2.node))

    def subtract(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise subtraction of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.subtract(t1.node, t2.node))

    def multiply(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise multiplication of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.multiply(t1.node, t2.node))

    def divide(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise division of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.divide(t1.node, t2.node))

    def equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise equality comparison of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.equal(t1.node, t2.node))

    def not_equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise inequality comparison of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.not_equal(t1.node, t2.node))

    def less(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise less-than comparison of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.less(t1.node, t2.node))

    def less_equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise less-than-or-equal comparison of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.less_equal(t1.node, t2.node))

    def greater(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise greater-than comparison of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.greater(t1.node, t2.node))

    def greater_equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise greater-than-or-equal comparison of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.greater_equal(t1.node, t2.node))

    def logical_and(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise logical AND of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.logical_and(t1.node, t2.node))

    def logical_or(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise logical OR of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.logical_or(t1.node, t2.node))

    def logical_xor(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise logical XOR of two tensors."""
        ensure_same_basis(self.basis, t1.space.basis)
        ensure_same_basis(self.basis, t2.space.basis)
        return self.new_tensor(graph.logical_xor(t1.node, t2.node))

    def logical_not(self, t: Tensor[B]) -> Tensor[B]:
        """Element-wise logical NOT of a tensor."""
        ensure_same_basis(self.basis, t.space.basis)
        return self.new_tensor(graph.logical_not(t.node))
