"""
Continuous Distributions
========================

This package defines the probability Density in terms of
the ScipyDistribution protocol. The Density class implements the
DensityOrMass protocol defined by the `generalized` module.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

import math
import numpy as np


@runtime_checkable
class ScipyDistribution(Protocol):
    """Protocol abstracting over continuous Scipy distributions."""

    def pdf(
        self,
        x: float | np.ndarray,
        **kwargs,
    ) -> float | np.ndarray: ...

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        **kwargs,
    ) -> float | np.ndarray: ...


class Density:
    """Models a probability density function.

    Args:
        dist: The underlying continuous distribution.
    """

    def __init__(self, dist: ScipyDistribution) -> None:
        self.dist = dist

    def support_size(self) -> float | int:
        """Implements DensityOrMass.support_size."""
        return math.inf

    def sample(self, k: int) -> list[float]:
        """Implements DensityOrMass.sample."""
        if k < 0:
            raise ValueError("k must be a zero or positive integer")
        rvs = self.dist.rvs(size=k)
        return [float(r) for r in rvs] if isinstance(rvs, np.ndarray) else [float(rvs)]

    def evaluate(self, x: float | list[float]) -> float | list[float]:
        """Implements DensityOrMass.evaluate."""
        ys = self.dist.pdf(np.asarray(x))
        return [float(y) for y in ys] if isinstance(ys, np.ndarray) else float(ys)
