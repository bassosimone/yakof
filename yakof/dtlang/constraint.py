"""
Constraints.
===========

This module defines constraint types used in the digital twins model
framework. Constraints represent relationships between resource usage and
available capacity, forming the core of the sustainability assessment.

Constraints can work with both deterministic capacities (represented as
tensors) and probabilistic capacities (represented by cumulative distribution
functions). When evaluated, constraints determine whether resource usage
remains within acceptable bounds relative to capacity.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from . import geometry


@runtime_checkable
class CumulativeDistribution(Protocol):
    """Protocol matching scipy.stats distributions interface."""

    def cdf(self, x: float | np.ndarray, *args, **kwds) -> float | np.ndarray: ...


class Constraint:
    """Represents a constraint between resource usage and capacity.

    A constraint associates a resource usage tensor with a capacity, which can be
    either a fixed tensor or a probabilistic distribution. When evaluated, the
    constraint indicates whether the usage is within acceptable bounds relative
    to the capacity.

    Args:
        usage: A tensor representing resource consumption
        capacity: Either a tensor or a distribution representing available capacity
        name: Optional constraint name
    """

    def __init__(
        self,
        usage: geometry.Tensor,
        capacity: geometry.Tensor | CumulativeDistribution,
        name: str = "",
    ) -> None:
        self.usage = usage
        self.capacity = capacity
        self.name = name
