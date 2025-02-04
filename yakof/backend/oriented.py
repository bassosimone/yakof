"""
Orientation-aware computation graph
===================================

This module provides orientation-aware tensor operations through Field objects that
act both as factories and containers for tensors of a specific orientation.

Example usage:

    >>> horizontal = Field[Horizontal]()
    >>> x = horizontal.constant(1.0)
    >>> y = horizontal.constant(2.0)
    >>> z = x + y  # OK - same orientation

    >>> vertical = Field[Vertical]()
    >>> w = vertical.constant(3.0)
    >>> bad = x + w  # Type error - cannot mix orientations

The Field class serves both as a factory for creating tensors with a specific
orientation and as a container to store and manage these tensors. All operations
preserve orientation type safety, ensuring that computations only occur between
tensors of the same orientation.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from typing import Dict, Generic, Protocol, runtime_checkable
from . import graph

# Basic types inherited from graph module
Shape = graph.Shape
Axis = graph.Axis
ScalarValue = graph.ScalarValue
DType = graph.DType


class Tensor[O]:
    """
    A tensor with orientation information.

    This class wraps a graph.Tensor with a type parameter that tracks its
    orientation. The orientation is purely a compile-time concept used
    to prevent mixing tensors with different orientations in computations.

    Type parameter:
        O: A type used as a marker for the tensor's orientation.
           Typically an empty class like Horizontal or Vertical.
    """

    def __init__(self, tensor: graph.Tensor):
        self.t = tensor

    def __hash__(self) -> int:
        return hash(self.t)

    def get_description(self) -> str:
        return self.t.description

    @property
    def dtype(self) -> DType:
        return self.t.dtype

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.t.name}')"

    @classmethod
    def ensure_tensor(cls, value: Tensor[O] | ScalarValue) -> Tensor[O]:
        """Convert a scalar value to an oriented tensor if it isn't one already."""
        if isinstance(value, (float, int, bool)):
            return cls(graph.constant(value))
        return value

    # Arithmetic operators
    def __add__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.add(self.t, self.ensure_tensor(other).t))

    def __radd__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.add(self.ensure_tensor(other).t, self.t))

    def __sub__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.subtract(self.t, self.ensure_tensor(other).t))

    def __rsub__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.subtract(self.ensure_tensor(other).t, self.t))

    def __mul__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.multiply(self.t, self.ensure_tensor(other).t))

    def __rmul__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.multiply(self.ensure_tensor(other).t, self.t))

    def __truediv__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.divide(self.t, self.ensure_tensor(other).t))

    def __rtruediv__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.divide(self.ensure_tensor(other).t, self.t))

    # Comparison operators
    def __eq__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:  # type: ignore
        return type(self)(graph.equal(self.t, self.ensure_tensor(other).t))

    def __ne__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:  # type: ignore
        return type(self)(graph.not_equal(self.t, self.ensure_tensor(other).t))

    def __lt__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.less(self.t, self.ensure_tensor(other).t))

    def __le__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.less_equal(self.t, self.ensure_tensor(other).t))

    def __gt__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.greater(self.t, self.ensure_tensor(other).t))

    def __ge__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.greater_equal(self.t, self.ensure_tensor(other).t))

    # Logical operators
    def __and__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.logical_and(self.t, self.ensure_tensor(other).t))

    def __rand__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.logical_and(self.ensure_tensor(other).t, self.t))

    def __or__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.logical_or(self.t, self.ensure_tensor(other).t))

    def __ror__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.logical_or(self.ensure_tensor(other).t, self.t))

    def __xor__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.logical_xor(self.t, self.ensure_tensor(other).t))

    def __rxor__(self, other: Tensor[O] | ScalarValue) -> Tensor[O]:
        return type(self)(graph.logical_xor(self.ensure_tensor(other).t, self.t))

    def __invert__(self) -> Tensor[O]:
        return type(self)(graph.logical_not(self.t))


@runtime_checkable
class TensorRegistry(Protocol):
    """A registry for tensors keeping track of their insertion order.

    The tensor passed to append_tensor SHOULD have its name set."""

    def append_tensor(self, tensor: graph.Tensor) -> None: ...


class TensorSpace[O]:
    """
    Acts as both a factory and container for tensors of a specific orientation.

    The tensor space ensures type safety by only allowing operations between tensors
    of the same orientation. It can also store tensors as attributes for
    convenient access and management.

    Type parameter:
        O: A type used as a marker for the orientation of all tensors in this space.
           Typically an empty class named, e.g., Horizontal or Vertical.
    """

    def __init__(self, registry: TensorRegistry):
        self._registry = registry

    def __setattr__(self, name: str, value: Tensor[O]):
        if name.startswith("_"):
            # Use parent class's __setattr__ for private attributes
            super().__setattr__(name, value)
            return
        if isinstance(value, graph.Tensor):
            value = Tensor[O](value)
        value.t.name = name
        self._registry.append_tensor(value.t)  # AFTER value.t.name
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Tensor[O]:
        raise AttributeError(f"{type(self)} has no attribute {name}")

    # Tensor operations
    def where(
        self,
        condition: Tensor[O] | ScalarValue,
        x: Tensor[O] | ScalarValue,
        y: Tensor[O] | ScalarValue,
    ) -> Tensor[O]:
        """Select elements based on condition."""
        condition = Tensor[O].ensure_tensor(condition)
        x = Tensor[O].ensure_tensor(x)
        y = Tensor[O].ensure_tensor(y)
        return Tensor[O](graph.where(condition.t, x.t, y.t))

    def multi_clause_where(
        self, *cases: tuple[Tensor[O] | ScalarValue, Tensor[O] | ScalarValue]
    ) -> Tensor[O]:
        """Conditional expression with multiple clauses."""
        wrapped_cases = [
            (Tensor[O].ensure_tensor(cond).t, Tensor[O].ensure_tensor(val).t)
            for cond, val in cases
        ]
        return Tensor[O](graph.multi_clause_where(*wrapped_cases))

    # Distribution functions
    def uniform_cdf(
        self,
        x: Tensor[O] | ScalarValue,
        loc: Tensor[O] | ScalarValue = 0.0,
        scale: Tensor[O] | ScalarValue = 1.0,
    ) -> Tensor[O]:
        """Compute uniform distribution CDF."""
        return Tensor[O](
            graph.uniform_cdf(
                Tensor[O].ensure_tensor(x).t,
                Tensor[O].ensure_tensor(loc).t,
                Tensor[O].ensure_tensor(scale).t,
            )
        )

    def normal_cdf(
        self,
        x: Tensor[O] | ScalarValue,
        loc: Tensor[O] | ScalarValue = 0.0,
        scale: Tensor[O] | ScalarValue = 1.0,
    ) -> Tensor[O]:
        """Compute normal distribution CDF."""
        return Tensor[O](
            graph.normal_cdf(
                Tensor[O].ensure_tensor(x).t,
                Tensor[O].ensure_tensor(loc).t,
                Tensor[O].ensure_tensor(scale).t,
            )
        )

    def maximum(
        self,
        x: Tensor[O] | ScalarValue,
        y: Tensor[O] | ScalarValue,
    ) -> Tensor[O]:
        """Element-wise maximum."""
        return Tensor[O](
            graph.maximum(
                Tensor[O].ensure_tensor(x).t,
                Tensor[O].ensure_tensor(y).t,
            )
        )

    def exp(self, x: Tensor[O] | ScalarValue) -> Tensor[O]:
        """Element-wise exponential."""
        return Tensor[O](graph.exp(Tensor[O].ensure_tensor(x).t))

    def power(
        self, x: Tensor[O] | ScalarValue, y: Tensor[O] | ScalarValue
    ) -> Tensor[O]:
        """Element-wise power function."""
        return Tensor[O](
            graph.power(Tensor[O].ensure_tensor(x).t, Tensor[O].ensure_tensor(y).t)
        )

    def pow(self, x: Tensor[O] | ScalarValue, y: Tensor[O] | ScalarValue) -> Tensor[O]:
        """Alias for power for compatibility with NumPy."""
        return self.power(x, y)

    def log(self, tensor: Tensor[O] | ScalarValue) -> Tensor[O]:
        """Element-wise logarithm while preserving orientation."""
        return Tensor[O](graph.log(Tensor[O].ensure_tensor(tensor).t))
