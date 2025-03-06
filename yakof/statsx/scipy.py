"""
Wrappers Around SciPy
=====================

This module defines wrappers around SciPy distributions that
simplify using them in a type-safe way. Users should not typically
import from this module directly, rather, they should instead
use the top-level statsx package.
"""

from scipy import stats
from typing import Any, Protocol, Sequence, cast, runtime_checkable

import numpy as np


@runtime_checkable
class ScipyRVContinuous(Protocol):
    """Protocol matching scipy.stats.rv_continuous.

    Methods:
        pdf: Probability density function of the distribution.
        rvs: Random variates sampling from the distribution.
    """

    def pdf(self, x: float | np.ndarray, **kwargs) -> float | np.ndarray: ...

    def rvs(
        self, size: int | tuple[int, ...] | None = None, **kwargs
    ) -> float | np.ndarray: ...


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
    """Create a uniform distribution over [loc, loc + scale].

    Args:
        loc: The lower bound (start) of the distribution.
        scale: The width (difference between upper and lower bounds).

    Returns:
        A uniform distribution compatible with the ScipyRVContinuous protocol.
    """
    return scipy_rv_continuous_cast(stats.uniform(loc=loc, scale=scale))


def scipy_normal(loc: float = 0.0, scale: float = 1.0) -> ScipyRVContinuous:
    """Create a normal distribution with the given location and scale.

    Args:
        loc: Mean ("center") of the distribution (default: 0.0)
        scale: Standard deviation (spread) of the distribution (default: 1.0)

    Returns:
        A normal distribution compatible with the ScipyRVContinuous protocol
    """
    return scipy_rv_continuous_cast(stats.norm(loc=loc, scale=scale))
