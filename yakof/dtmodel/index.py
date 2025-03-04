from typing import Protocol, runtime_checkable

import numpy as np

from ..frontend import graph

from . import geometry


@runtime_checkable
class Initializer(Protocol):
    def rvs(self, size: int = 1) -> np.ndarray: ...


class _ScalarInitializer:
    def __init__(self, value: graph.Scalar) -> None:
        self.value = value

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.asarray(self.value)


class Placeholder(geometry.ComputationTensor):
    def __init__(self, node: graph.Node, initializer: Initializer) -> None:
        super().__init__(geometry.ComputationSpace, node)
        self.initializer = initializer


def construct(name: str, value: graph.Scalar | Initializer) -> Placeholder:
    if isinstance(value, graph.Scalar):
        return Placeholder(graph.placeholder(name), _ScalarInitializer(value))
    return Placeholder(graph.placeholder(name), value)
