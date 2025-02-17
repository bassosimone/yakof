"""
Abstract Tensor Operations
==========================

This modules defines tensors spaces, tensors within spaces, and mapping
operations to convert tensors across spaces. It builds on top of the
computation graph (yakof.frontend.graph) to provide a type-safe interface
for working with tensors in different spaces.

The module provides four main abstractions:

1. TensorSpace[B]: A space of tensors with a given basis.
   Provides mathematical operations such as exp, log, and power.

2. Tensor[B]: A tensor with associated basis vectors.
   Supports arithmetic, comparison, and logical operations.

3. Basis: A protocol defining the required attributes for basis vectors.

4. TensorMap[A, B]: A structure-preserving map between tensor spaces.
   Supports expansion into higher dimensions and projection to lower dimensions.

The type parameters ensure that operations between tensors are only possible
when they share the same context and basis. TensorMap provides type-safe
transformations between different bases.

Type System Design
------------------

The type system uses generics to enforce:

1. Basis compatibility:
   - Operations only between tensors with same basis
   - Type-safe transformations between spaces
   - Compile-time detection of dimension mismatches

2. Context preservation:
   - Morphisms preserve structural properties
   - Clear distinction between different tensor spaces

This design enables catching errors at compile time.

See Also
--------

yakof.frontend
    Provides an overview for the tensor language frontend.

yakof.frontend.bases
    Basis definitions for tensor spaces to be used along with yakof.frontend.abstract.

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

    def __init__(self, t: graph.Node) -> None:
        self.t = t

    @property
    def name(self) -> str:
        return self.t.name

    @name.setter
    def name(self, value: str) -> None:
        self.t.name = value

    def __hash__(self) -> int:
        return hash(self.t)  # hashing by identity

    # Arithmetic operators
    def __add__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.add(self.t, _ensure_tensor[B]()(other).t))

    def __radd__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.add(_ensure_tensor[B]()(other).t, self.t))

    def __sub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.subtract(self.t, _ensure_tensor[B]()(other).t))

    def __rsub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.subtract(_ensure_tensor[B]()(other).t, self.t))

    def __mul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.multiply(self.t, _ensure_tensor[B]()(other).t))

    def __rmul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.multiply(_ensure_tensor[B]()(other).t, self.t))

    def __truediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.divide(self.t, _ensure_tensor[B]()(other).t))

    def __rtruediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.divide(_ensure_tensor[B]()(other).t, self.t))

    # Comparison operators
    def __eq__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return type(self)(graph.equal(self.t, _ensure_tensor[B]()(other).t))

    def __ne__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return type(self)(graph.not_equal(self.t, _ensure_tensor[B]()(other).t))

    def __lt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.less(self.t, _ensure_tensor[B]()(other).t))

    def __le__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.less_equal(self.t, _ensure_tensor[B]()(other).t))

    def __gt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.greater(self.t, _ensure_tensor[B]()(other).t))

    def __ge__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.greater_equal(self.t, _ensure_tensor[B]()(other).t))

    # Logical operators
    def __and__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_and(self.t, _ensure_tensor[B]()(other).t))

    def __rand__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_and(_ensure_tensor[B]()(other).t, self.t))

    def __or__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_or(self.t, _ensure_tensor[B]()(other).t))

    def __ror__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_or(_ensure_tensor[B]()(other).t, self.t))

    def __xor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_xor(self.t, _ensure_tensor[B]()(other).t))

    def __rxor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_xor(_ensure_tensor[B]()(other).t, self.t))

    def __invert__(self) -> Tensor[B]:
        return type(self)(graph.logical_not(self.t))


class TensorSpace[B]:
    """A space of tensors with associated basis vectors.

    Type Parameters:
        B: Type of the basis vectors.
    """

    # TODO(bassosimone): TensorSpace should contain all the operations
    # that are defined for graph and don't change the tensor basis.

    def __init__(self, b: type[B]) -> None:
        # We only need the constructor to receive a type for the
        # type system to automatically assign a type to the instance
        # and to have a similar instantiation pattern as TensorMap
        # but we have no use for the type itself.
        pass

    def placeholder(
        self, name: str = "", default_value: graph.Scalar | None = None
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
        return type(t)(graph.exp(t.t))

    def power(self, t: Tensor[B], exp: Tensor[B]) -> Tensor[B]:
        """Raise tensor to the power of another tensor."""
        return type(t)(graph.power(t.t, exp.t))

    def log(self, t: Tensor[B]) -> Tensor[B]:
        """Compute the natural logarithm of a tensor."""
        return type(t)(graph.log(t.t))

    def maximum(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Compute element-wise maximum of two tensors."""
        return type(t1)(graph.maximum(t1.t, t2.t))

    def where[
        C
    ](self, cond: Tensor[C], then: Tensor[B], otherwise: Tensor[B]) -> Tensor[B]:
        """Select elements based on condition."""
        return type(then)(graph.where(cond.t, then.t, otherwise.t))

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
                clauses=[(cond.t, value.t) for cond, value in clauses],
                default_value=_ensure_tensor[B]()(default_value).t,
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
                x.t, _ensure_tensor[B]()(loc).t, _ensure_tensor[B]()(scale).t
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
                x.t, _ensure_tensor[B]()(loc).t, _ensure_tensor[B]()(scale).t
            )
        )

    def tracepoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a tracepoint for the current tensor inside the computation graph."""
        return Tensor[B](graph.tracepoint(t.t))

    def breakpoint(self, t: Tensor[B]) -> Tensor[B]:
        """Inserts a breakpoint for the current tensor inside the computation graph."""
        return Tensor[B](graph.breakpoint(t.t))


@runtime_checkable
class Basis(Protocol):
    """Protocol defining required attributes for basis vectors.

    Methods:
        axis: Return the axis of the basis vector.
    """

    @staticmethod
    def axis() -> graph.Axis: ...


class TensorMap[A: Basis, B: Basis]:
    """A structure-preserving map between tensor spaces.

    Type Parameters:
        A: Source basis type for expansion / target for projection
        B: Target basis type for expansion / source for projection
    """

    def __init__(self, a: type[A], b: type[B]) -> None:
        self.a = a
        self.b = b

    def expand_dims(self, t: Tensor[A]) -> Tensor[B]:
        """Expand tensor into higher-dimensional space."""
        return Tensor[B](graph.expand_dims(t.t, axis=self.b.axis()))

    def project_using_mean(self, t: Tensor[B]) -> Tensor[A]:
        """Project tensor to lower dimension using mean reduction."""
        return Tensor[A](graph.reduce_mean(t.t, axis=self.a.axis()))

    def project_using_sum(self, t: Tensor[B]) -> Tensor[A]:
        """Project tensor to lower dimension using sum reduction."""
        return Tensor[A](graph.reduce_sum(t.t, axis=self.a.axis()))
