"""Sympy-like operators for creating symbolic expressions."""

from ..frontend import graph
from .symbol import SymbolValue


def _ensure_node(value: graph.Node | SymbolValue | graph.Scalar) -> graph.Node:
    return (
        value.node
        if isinstance(value, SymbolValue)
        else graph.constant(value) if isinstance(value, graph.Scalar) else value
    )


def Eq(
    lhs: graph.Node | SymbolValue | graph.Scalar,
    rhs: graph.Node | SymbolValue | graph.Scalar,
) -> graph.Node:
    """Create an equality expression between nodes and/or symbols."""
    return graph.equal(_ensure_node(lhs), _ensure_node(rhs))
