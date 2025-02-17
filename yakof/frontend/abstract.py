"""
Abstract Tensor Operations
==========================

This modules defines tensors spaces, tensors within spaces, and mapping
operations to convert tensors across spaces. It builds on top of the
computation graph (yakof.frontend.graph) to provide a type-safe interface
for working with tensors in different spaces.

The module provides these abstractions:

1. TensorSpace[B]: A space of tensors with a given basis.
   Provides mathematical operations such as exp, log, and power.

2. Tensor[B]: A tensor with associated basis vectors.
   Supports arithmetic, comparison, and logical operations.

The type parameters ensure that operations between tensors are only possible
when they share the same context and basis.

Type System Design
------------------

The type system uses generics to enforce:

1. Basis compatibility:
   - Operations only between tensors with same basis
   - Compile-time detection of dimension mismatches

2. Context preservation:
   - Clear distinction between different tensor spaces

This design enables catching errors at compile time.

See Also
--------

yakof.frontend
    Provides an overview for the tensor language frontend.

yakof.frontend.bases
    Basis definitions and base transformation operations to be
    used along with yakof.frontend.abstract.

yakof.frontend.graph
    Computation graph building.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from . import graph


class _ensure_tensor[B]:
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

    @property
    def name(self) -> str:
        return self.node.name

    @name.setter
    def name(self, value: str) -> None:
        self.node.name = value

    def __hash__(self) -> int:
        return hash(self.node)  # hashing by identity

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

    # TODO(bassosimone): TensorSpace should contain all the operations
    # that are defined for graph and don't change the tensor basis.

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

        Regardless, a placeholder with no name cannot be used in the graph.
        """
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

    def normal_cdf(
        self,
        x: Tensor[B],
        loc: Tensor[B] | graph.Scalar = 0.0,
        scale: Tensor[B] | graph.Scalar = 1.0,
    ) -> Tensor[B]:
        """Compute normal distribution CDF."""
        return Tensor[B](
            graph.normal_cdf(
                x.node, _ensure_tensor[B]()(loc).node, _ensure_tensor[B]()(scale).node
            )
        )

    def uniform_cdf(
        self,
        x: Tensor[B],
        loc: Tensor[B] | graph.Scalar = 0.0,
        scale: Tensor[B] | graph.Scalar = 1.0,
    ) -> Tensor[B]:
        """Compute uniform distribution CDF."""
        return Tensor[B](
            graph.uniform_cdf(
                x.node, _ensure_tensor[B]()(loc).node, _ensure_tensor[B]()(scale).node
            )
        )

    def tracepoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a tracepoint for the current tensor inside the computation graph."""
        return Tensor[B](graph.tracepoint(t.node))

    def breakpoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a breakpoint for the current tensor inside the computation graph."""
        return Tensor[B](graph.breakpoint(t.node))
