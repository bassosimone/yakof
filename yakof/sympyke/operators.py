"""
...
"""

from ..frontend import graph


def Eq(lhs: graph.Node, rhs: graph.Node) -> graph.Node:
    return graph.equal(lhs, rhs)
