"""
Index-based language
====================

..........
"""

from __future__ import annotations

from ..frontend import graph


class Index:
    def __init__(self, node: graph.Node) -> None:
        self.node = node

    # Implementation of autonaming.Namer protocol
    def implements_namer(self) -> None:
        pass
    @property
    def name(self) -> str:
        return self.node.name
    @name.setter
    def name(self, value: str) -> None:
        self.node.name = value

    # Required to store Indexes inside sets and dictionaries
    def __hash__(self) -> int:
        return hash(self.node)

    # Basic arithmetic operations
    def __add__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.add(self.node, _ensure_index(other).node))

    def __radd__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__add__(self)

    def __sub__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.subtract(self.node, _ensure_index(other).node))

    def __rsub__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__sub__(self)

    def __mul__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.multiply(self.node, _ensure_index(other).node))

    def __rmul__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__mul__(self)

    def __truediv__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.divide(self.node, _ensure_index(other).node))

    def __rtruediv__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__truediv__(self)

    def __pow__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.power(self.node, _ensure_index(other).node))

    def __rpow__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__pow__(self)

    # Comparison operations
    def __eq__(self, other: graph.Scalar | Index) -> Index:  # type: ignore
        return Index(graph.equal(self.node, _ensure_index(other).node))

    def __ne__(self, other: graph.Scalar | Index) -> Index:  # type: ignore
        return Index(graph.not_equal(self.node, _ensure_index(other).node))

    def __lt__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.less(self.node, _ensure_index(other).node))

    def __le__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.less_equal(self.node, _ensure_index(other).node))

    def __gt__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.greater(self.node, _ensure_index(other).node))

    def __ge__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.greater_equal(self.node, _ensure_index(other).node))

    # Logical operations
    def __and__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.logical_and(self.node, _ensure_index(other).node))

    def __rand__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__and__(self)

    def __or__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.logical_or(self.node, _ensure_index(other).node))

    def __ror__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__or__(self)

    def __xor__(self, other: graph.Scalar | Index) -> Index:
        return Index(graph.logical_xor(self.node, _ensure_index(other).node))

    def __rxor__(self, other: graph.Scalar | Index) -> Index:
        return _ensure_index(other).__xor__(self)

    def __invert__(self) -> Index:
        return Index(graph.logical_not(self.node))

    # Mathematical operations
    def exp(self) -> Index:
        """Compute exponential."""
        return Index(graph.exp(self.node))

    def log(self) -> Index:
        """Compute natural logarithm."""
        return Index(graph.log(self.node))

    def maximum(self, other: graph.Scalar | Index) -> Index:
        """Compute element-wise maximum."""
        return Index(graph.maximum(self.node, _ensure_index(other).node))

    # Debug support
    def tracepoint(self) -> Index:
        """Insert tracepoint for debugging."""
        return Index(graph.tracepoint(self.node))

    def breakpoint(self) -> Index:
        """Insert breakpoint for debugging."""
        return Index(graph.breakpoint(self.node))


def _ensure_index(other: graph.Scalar | Index) -> Index:
    """Convert value to Index if needed."""
    if isinstance(other, Index):
        return other
    return Index(graph.constant(other))
