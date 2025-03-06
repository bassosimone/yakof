"""
Wrappers Around SciPy
=====================

This module defines wrappers around SciPy distributions that
simplify using them in a type-safe way.
"""

from scipy import stats
from typing import Any, Protocol, Sequence, cast, runtime_checkable

import numpy as np


@runtime_checkable
class ScipyRVContinuous(Protocol):
    """Protocol matching scipy.stats rv_continuous.

    Methods:
        pdf: Probability density function of the distribution.
        rvs: Random variates with specified parameters.
    """

    def pdf(self, x: float | Sequence[float]) -> float | np.ndarray: ...
    def rvs(self, size: int | None = None) -> float | np.ndarray: ...


def scipy_rv_continuous_cast(dist: Any) -> ScipyRVContinuous:
    """Casts a distribution returned by scipy.stats into an RVContinuous.

    This function helps bridge the gap between SciPy's complex type system
    and our simplified protocol interface. It should be used with distributions
    that have pdf() and rvs() methods.
    """
    if not hasattr(dist, "pdf") or not hasattr(dist, "rvs"):
        raise TypeError("distribution must have the pdf and rvs methods")
    return cast(ScipyRVContinuous, dist)


def scipy_uniform(loc: float, scale: float) -> ScipyRVContinuous:
    """Create a uniform distribution with the given loc and scale."""
    return scipy_rv_continuous_cast(stats.uniform(loc, scale))


def scipy_normal(loc: float = 0.0, scale: float = 1.0) -> ScipyRVContinuous:
    """Create a normal distribution with the given location and scale.

    Args:
        loc: Mean ("center") of the distribution (default: 0.0)
        scale: Standard deviation (spread) of the distribution (default: 1.0)

    Returns:
        A normal distribution compatible with the ScipyRVContinuous protocol
    """
    return scipy_rv_continuous_cast(stats.norm(loc=loc, scale=scale))
