"""
Abstract Tensor Operations
==========================

This modules defines tensors spaces and tensors within spaces. It builds on
top of the computation graph (`graph.py`) to provide a type-safe interface for
working with tensors in different tensor spaces.

The module provides these abstractions:

1. TensorSpace[B]: A space of tensors with a given basis.
   Provides mathematical operations such as exp, log, and power.

2. Tensor[B]: A tensor with associated basis vectors.
   Supports arithmetic, comparison, and logical operations.

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

1. Objects are tensor spaces (TensorSpace[B])
2. Morphisms are, in general, structure-preserving maps between tensor spaces
3. Endomorphisms are operations within a tensor space that preserve its structure

However, note that this module only implements endomorphisms. We implement
space-changing morphisms in the `morphisms.py` module instead.

Key categorical properties:

1. Identity: Each tensor space has identity operations.
2. Composition: Operations can be composed while preserving types.
3. Associativity: Operations composition is associative.

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

Thus, we can detect errors at compile time.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

from . import graph


class _ensure_tensor[B]:
    """Helper class to ensure that a scalar is converted to a tensor."""

    def __call__(self, t: Tensor[B] | graph.Scalar) -> Tensor[B]:
        if isinstance(t, graph.Scalar):
            t = Tensor[B](graph.constant(t))
        return t


class Tensor[B]:
    """A tensor with associated basis vectors.

    Type Parameters:
        B: Type of the basis vectors.
    """

    def __init__(self, node: graph.Node) -> None:
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
        return type(self)(graph.add(self.node, _ensure_tensor[B]()(other).node))

    def __radd__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.add(_ensure_tensor[B]()(other).node, self.node))

    def __sub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.subtract(self.node, _ensure_tensor[B]()(other).node))

    def __rsub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.subtract(_ensure_tensor[B]()(other).node, self.node))

    def __mul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.multiply(self.node, _ensure_tensor[B]()(other).node))

    def __rmul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.multiply(_ensure_tensor[B]()(other).node, self.node))

    def __truediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.divide(self.node, _ensure_tensor[B]()(other).node))

    def __rtruediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.divide(_ensure_tensor[B]()(other).node, self.node))

    # Comparison operators
    #
    # See corresponding comment in graph.py
    def __eq__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return type(self)(graph.equal(self.node, _ensure_tensor[B]()(other).node))

    def __ne__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return type(self)(graph.not_equal(self.node, _ensure_tensor[B]()(other).node))

    def __lt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.less(self.node, _ensure_tensor[B]()(other).node))

    def __le__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.less_equal(self.node, _ensure_tensor[B]()(other).node))

    def __gt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.greater(self.node, _ensure_tensor[B]()(other).node))

    def __ge__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(
            graph.greater_equal(self.node, _ensure_tensor[B]()(other).node)
        )

    # Logical operators
    def __and__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_and(self.node, _ensure_tensor[B]()(other).node))

    def __rand__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_and(_ensure_tensor[B]()(other).node, self.node))

    def __or__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_or(self.node, _ensure_tensor[B]()(other).node))

    def __ror__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_or(_ensure_tensor[B]()(other).node, self.node))

    def __xor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_xor(self.node, _ensure_tensor[B]()(other).node))

    def __rxor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_xor(_ensure_tensor[B]()(other).node, self.node))

    def __invert__(self) -> Tensor[B]:
        return type(self)(graph.logical_not(self.node))


class TensorSpace[B]:
    """A space of tensors with associated basis vectors.

    Type Parameters:
        B: Type of the basis vectors.
    """

    def placeholder(
        self,
        name: str = "",
        default_value: graph.Scalar | None = None,
    ) -> Tensor[B]:
        """Creates a placeholder tensor.

        The name parameter is optional to allow using autonaming.context():

            >>> with autonaming.context():
            ...     x = space.placeholder()  # automatically named 'x'

        But must be explicitly provided otherwise:

            >>> y = space.placeholder("y")  # explicitly named

        Failing to name a tensor would cause the grap evaluation to fail. We highly
        recommend using autonaming to provide a name to tensors.
        """
        # TODO(bassosimone): perhaps autonaming should be a higher-level feature? We should
        # decide whether do to this *before* merging this code into the dt-model repo.
        return Tensor[B](graph.placeholder(name, default_value))

    def constant(self, value: graph.Scalar, name: str = "") -> Tensor[B]:
        """Creates a constant tensor."""
        return Tensor[B](graph.constant(value, name))

    def exp(self, t: Tensor[B]) -> Tensor[B]:
        """Compute the exponential of a tensor."""
        return type(t)(graph.exp(t.node))

    def power(self, t: Tensor[B], exp: Tensor[B]) -> Tensor[B]:
        """Raise tensor to the power of another tensor."""
        return type(t)(graph.power(t.node, exp.node))

    def log(self, t: Tensor[B]) -> Tensor[B]:
        """Compute the natural logarithm of a tensor."""
        return type(t)(graph.log(t.node))

    def maximum(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Compute element-wise maximum of two tensors."""
        return type(t1)(graph.maximum(t1.node, t2.node))

    def where[
        C
    ](self, cond: Tensor[C], then: Tensor[B], otherwise: Tensor[B]) -> Tensor[B]:
        """Select elements based on condition."""
        return type(then)(graph.where(cond.node, then.node, otherwise.node))

    def multi_clause_where[
        C
    ](
        self,
        clauses: Sequence[tuple[Tensor[C], Tensor[B]]],
        default_value: Tensor[B] | graph.Scalar,
    ) -> Tensor[B]:
        """Select elements based on multiple conditions."""
        return Tensor[B](
            graph.multi_clause_where(
                clauses=[(cond.node, value.node) for cond, value in clauses],
                default_value=_ensure_tensor[B]()(default_value).node,
            )
        )

    def tracepoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a tracepoint for the current tensor inside the computation graph."""
        return Tensor[B](graph.tracepoint(t.node))

    def breakpoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a breakpoint for the current tensor inside the computation graph."""
        return Tensor[B](graph.breakpoint(t.node))

    # Additional shape/structure preserving operations
    def add(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise addition of two tensors."""
        return type(t1)(graph.add(t1.node, t2.node))

    def subtract(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise subtraction of two tensors."""
        return type(t1)(graph.subtract(t1.node, t2.node))

    def multiply(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise multiplication of two tensors."""
        return type(t1)(graph.multiply(t1.node, t2.node))

    def divide(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise division of two tensors."""
        return type(t1)(graph.divide(t1.node, t2.node))

    def equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise equality comparison of two tensors."""
        return type(t1)(graph.equal(t1.node, t2.node))

    def not_equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise inequality comparison of two tensors."""
        return type(t1)(graph.not_equal(t1.node, t2.node))

    def less(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise less-than comparison of two tensors."""
        return type(t1)(graph.less(t1.node, t2.node))

    def less_equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise less-than-or-equal comparison of two tensors."""
        return type(t1)(graph.less_equal(t1.node, t2.node))

    def greater(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise greater-than comparison of two tensors."""
        return type(t1)(graph.greater(t1.node, t2.node))

    def greater_equal(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise greater-than-or-equal comparison of two tensors."""
        return type(t1)(graph.greater_equal(t1.node, t2.node))

    def logical_and(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise logical AND of two tensors."""
        return type(t1)(graph.logical_and(t1.node, t2.node))

    def logical_or(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise logical OR of two tensors."""
        return type(t1)(graph.logical_or(t1.node, t2.node))

    def logical_xor(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Element-wise logical XOR of two tensors."""
        return type(t1)(graph.logical_xor(t1.node, t2.node))

    def logical_not(self, t: Tensor[B]) -> Tensor[B]:
        """Element-wise logical NOT of a tensor."""
        return type(t)(graph.logical_not(t.node))
