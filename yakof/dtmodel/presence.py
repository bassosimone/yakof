from typing import Sequence

from . import context, geometry

from ..frontend import graph


class Variable(geometry.ComputationTensor):
    def __init__(self, node: graph.Node, cvs: Sequence[context.Variable]) -> None:
        super().__init__(geometry.ComputationSpace, node)
        self.cvs = cvs


def construct(name: str, cvs: Sequence[context.Variable]) -> Variable:
    return Variable(graph.placeholder(name), cvs)
