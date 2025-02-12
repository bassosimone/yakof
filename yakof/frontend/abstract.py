"""
Abstract Tensor Operations
==========================

This module defines the abstract tensor operations and morphisms between
tensor spaces. It builds on top of the computation graph to provide a type-safe
interface for working with tensors in different spaces.

The module provides four main abstractions:

1. Tensor[B]: A tensor with associated basis vectors.
   Supports arithmetic, comparison, and logical operations.

2. TensorSpace[B]: A space of tensors with a given basis.
   Provides mathematical operations such as exp, log, and power.

3. Basis: A protocol defining the required attributes for basis vectors.

4. Morphism[A, B]: A structure-preserving map between tensor spaces.
   Supports expansion into higher dimensions and projection to lower dimensions.

The type parameters ensure that operations between tensors are only possible
when they share the same context and basis. Morphisms provide type-safe
transformations between different bases.

Type System Design
------------------

The type system uses generics to enforce:

1. Basis compatibility:
   - Operations only between tensors with same basis
   - Type-safe transformations between spaces
   - Compile-time detection of dimension mismatches

2. Context preservation:
   - Operations maintain tensor space context
   - Morphisms preserve structural properties
   - Clear distinction between different tensor spaces

This design enables catching errors at compile time rather
than runtime, while maintaining flexibility through morphisms.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from . import graph


class Tensor[B]:
    """A tensor with associated basis vectors.

    Type Parameters:
        B: Type of the basis vectors.
    """

    def __init__(self, t: graph.Node) -> None:
        self.t = t

    def __hash__(self) -> int:
        return hash(self.t)

    def _ensure_tensor(self, t: Tensor[B] | graph.Scalar) -> Tensor[B]:
        if isinstance(t, graph.Scalar):
            t = type(self)(graph.constant(t))
        return t

    # Arithmetic operators
    def __add__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.add(self.t, self._ensure_tensor(other).t))

    def __radd__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.add(self._ensure_tensor(other).t, self.t))

    def __sub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.subtract(self.t, self._ensure_tensor(other).t))

    def __rsub__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.subtract(self._ensure_tensor(other).t, self.t))

    def __mul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.multiply(self.t, self._ensure_tensor(other).t))

    def __rmul__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.multiply(self._ensure_tensor(other).t, self.t))

    def __truediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.divide(self.t, self._ensure_tensor(other).t))

    def __rtruediv__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.divide(self._ensure_tensor(other).t, self.t))

    # Comparison operators
    def __eq__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return type(self)(graph.equal(self.t, self._ensure_tensor(other).t))

    def __ne__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:  # type: ignore
        return type(self)(graph.not_equal(self.t, self._ensure_tensor(other).t))

    def __lt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.less(self.t, self._ensure_tensor(other).t))

    def __le__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.less_equal(self.t, self._ensure_tensor(other).t))

    def __gt__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.greater(self.t, self._ensure_tensor(other).t))

    def __ge__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.greater_equal(self.t, self._ensure_tensor(other).t))

    # Logical operators
    def __and__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_and(self.t, self._ensure_tensor(other).t))

    def __rand__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_and(self._ensure_tensor(other).t, self.t))

    def __or__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_or(self.t, self._ensure_tensor(other).t))

    def __ror__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_or(self._ensure_tensor(other).t, self.t))

    def __xor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_xor(self.t, self._ensure_tensor(other).t))

    def __rxor__(self, other: Tensor[B] | graph.Scalar) -> Tensor[B]:
        return type(self)(graph.logical_xor(self._ensure_tensor(other).t, self.t))

    def __invert__(self) -> Tensor[B]:
        return type(self)(graph.logical_not(self.t))


class TensorSpace[B]:
    """A space of tensors with a given context and basis.

    Type Parameters:
        B: Type of the basis vectors.
    """

    def __init__(self, b: type[B]) -> None:
        # We only need the constructor to receive a type for the
        # type system to automatically assign a type to the instance
        # and to have a similar instantiation pattern as Morphism.
        pass

    def placeholder(
        self, name: str, default_value: graph.Scalar | None = None
    ) -> Tensor[B]:
        """Creates a placeholder tensor."""
        return Tensor[B](graph.placeholder(name, default_value))

    def constant(self, value: graph.Scalar) -> Tensor[B]:
        """Creates a constant tensor."""
        return Tensor[B](graph.constant(value))

    def exp(self, t: Tensor[B]) -> Tensor[B]:
        """Compute exponential of tensor."""
        return type(t)(graph.exp(t.t))

    def power(self, t: Tensor[B], exp: Tensor[B]) -> Tensor[B]:
        """Raise tensor to power of another tensor."""
        return type(t)(graph.power(t.t, exp.t))

    def log(self, t: Tensor[B]) -> Tensor[B]:
        """Compute natural logarithm of tensor."""
        return type(t)(graph.log(t.t))

    def maximum(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
        """Compute element-wise maximum of two tensors."""
        return type(t1)(graph.maximum(t1.t, t2.t))

    def where[
        C: Basis
    ](self, cond: Tensor[C], then: Tensor[B], otherwise: Tensor[B],) -> Tensor[B]:
        """Select elements based on condition."""
        return type(then)(graph.where(cond.t, then.t, otherwise.t))

    def multi_clause_where[
        C: Basis
    ](self, clauses: list[tuple[Tensor[C], Tensor[B]]]) -> Tensor[B]:
        """Select elements based on multiple conditions."""
        return Tensor[B](
            graph.multi_clause_where(*((cond.t, value.t) for cond, value in clauses))
        )


@runtime_checkable
class Basis(Protocol):
    """Protocol defining required attributes for basis vectors.

    Methods:
        axis: Return the axis of the basis vector.
    """

    @staticmethod
    def axis() -> graph.Axis: ...


class Morphism[A: Basis, B: Basis]:
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
