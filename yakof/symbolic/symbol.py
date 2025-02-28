"""
Symbolic Graph Node
===================

This module provides symbolic computation capabilities on top of yakof's
frontend graph. The Symbol class supports:

1. Basic arithmetic operations (+, -, *, /)
2. Comparison operations (<, <=, >, >=, ==, !=)
3. Logical operations (&, |, ^, ~)
4. Mathematical operations (exp, log, etc)
5. Autonaming support

Examples:
    >>> x = Symbol("x")
    >>> y = Symbol("y", 2.0)
    >>> z = x * y + 1
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ..frontend import graph, autonaming


class Symbol:
    """Wraps a graph.Node to provide sympy-like interface."""

    def __init__(self, name: str = "", value: graph.Scalar | None = None) -> None:
        """Create a new symbol.

        Args:
            name: Symbol name (for placeholders)
            value: Optional constant value
        """
        if value is not None:
            self.node = graph.constant(value, name)
            print("ELLIOT: creating a constant node with value {value}")
        else:
            self.node = graph.placeholder(name, default_value=value)
            print(
                f"ELLIOT: creating a placeholder node with name {name} and value {value}"
            )

    def __hash__(self) -> int:
        return hash(self.node)

    @staticmethod
    def _wrap(node: graph.Node) -> Symbol:
        """Wrap an existing node in a Symbol."""
        sym = Symbol()
        sym.node = node
        return sym

    def _ensure_symbol(self, other: Any) -> Symbol:
        """Convert value to Symbol if needed."""
        if isinstance(other, Symbol):
            return other
        return Symbol("", other)

    # Basic arithmetic operations
    def __add__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.add(self.node, other.node))

    def __radd__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__add__(self)

    def __sub__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.subtract(self.node, other.node))

    def __rsub__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__sub__(self)

    def __mul__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.multiply(self.node, other.node))

    def __rmul__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__mul__(self)

    def __truediv__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.divide(self.node, other.node))

    def __rtruediv__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__truediv__(self)

    def __pow__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.power(self.node, other.node))

    def __rpow__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__pow__(self)

    # Comparison operations
    def __eq__(self, other: Any) -> Symbol:  # type: ignore
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.equal(self.node, other.node))

    def __ne__(self, other: Any) -> Symbol:  # type: ignore
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.not_equal(self.node, other.node))

    def __lt__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.less(self.node, other.node))

    def __le__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.less_equal(self.node, other.node))

    def __gt__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.greater(self.node, other.node))

    def __ge__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.greater_equal(self.node, other.node))

    # Logical operations
    def __and__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.logical_and(self.node, other.node))

    def __rand__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__and__(self)

    def __or__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.logical_or(self.node, other.node))

    def __ror__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__or__(self)

    def __xor__(self, other: Any) -> Symbol:
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.logical_xor(self.node, other.node))

    def __rxor__(self, other: Any) -> Symbol:
        return self._ensure_symbol(other).__xor__(self)

    def __invert__(self) -> Symbol:
        return Symbol._wrap(graph.logical_not(self.node))

    # Mathematical operations
    def exp(self) -> Symbol:
        """Compute exponential."""
        return Symbol._wrap(graph.exp(self.node))

    def log(self) -> Symbol:
        """Compute natural logarithm."""
        return Symbol._wrap(graph.log(self.node))

    def maximum(self, other: Any) -> Symbol:
        """Compute element-wise maximum."""
        other = self._ensure_symbol(other)
        return Symbol._wrap(graph.maximum(self.node, other.node))

    # Debug support
    def tracepoint(self) -> Symbol:
        """Insert tracepoint for debugging."""
        return Symbol._wrap(graph.tracepoint(self.node))

    def breakpoint(self) -> Symbol:
        """Insert breakpoint for debugging."""
        return Symbol._wrap(graph.breakpoint(self.node))

    # Required by yakof's autonaming
    def implements_namer(self) -> None:
        """Part of autonaming.Namer protocol."""
        pass

    @property
    def name(self) -> str:
        """Get symbol name."""
        return self.node.name

    @name.setter
    def name(self, value: str) -> None:
        """Set symbol name."""
        self.node.name = value
