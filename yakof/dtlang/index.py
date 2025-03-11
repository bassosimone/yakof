"""
Index Implementation
====================

This module implements the `Index` type, which represents either a constant
value or a placeholder for sampling from a random distribution. In either case,
we compile the Index into a graph.placeholder tensor in the XYZ space.
"""

from typing import Protocol, cast, runtime_checkable

import numpy as np

from ..frontend import graph

from . import geometry


@runtime_checkable
class Distribution(Protocol):
    """Protocol representing the distribution from which to sample the
    index value inside the ensemble space.

    Methods:
        rvs: Sample random variates from the distribution.
    """

    def rvs(self, size: int = 1) -> np.ndarray: ...


class _Constant:
    """Fake Distribution that just returns a constant value."""

    def __init__(self, value: graph.Scalar) -> None:
        self.value = value

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.asarray(self.value)


# Make sure that the _Constant type implements Distribution
_: Distribution = _Constant(0)


class Index(geometry.Tensor):
    """The Index is a geometry tensor representing either a constant value
    or a placeholder for sampling from a random distribution. In either case,
    the evaluator will need to use the `distribution` attribute to get the
    initial placeholder value through sampling.

    Args:
        name: The name of the index.
        value: The initial value of the index, which can be a constant or a
            distribution from which to sample in the ensemble space.
    """

    def __init__(
        self,
        name: str,
        value: graph.Scalar | Distribution,
    ) -> None:
        if isinstance(value, graph.Scalar):
            value = _Constant(value)
        super().__init__(geometry.space, graph.placeholder(name))
        self.distribution = cast(Distribution, value)
