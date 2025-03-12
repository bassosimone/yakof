"""
...
"""

from ..frontend import graph
from .symbol import SymbolValue

def Eq(lhs: graph.Node | SymbolValue, rhs: graph.Node | SymbolValue) -> graph.Node:
    return graph.equal(
        lhs.node if isinstance(lhs, SymbolValue) else lhs,
        rhs.node if isinstance(rhs, SymbolValue) else rhs,
    )
