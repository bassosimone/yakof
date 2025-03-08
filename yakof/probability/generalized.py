"""
Generalized Distributions
=========================

This module defines the DensityOrMass protocol that abstracts over
continuous and discrete distributions. It also provides factory functions
for creating several continuous and discrete distributions.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import (
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from scipy import stats

from .continuous import Density, ScipyDistribution
from .discrete import Mass

T = TypeVar("T")
"""Type alias for the distribution support."""


@runtime_checkable
class DensityOrMass(Generic[T], Protocol):
    """Protocol abstracting over continuous and discrete distributions.

    Methods:
        support_size: Returns the size of the support.
        sample: Generates a sample of size k.
        evaluate: Evaluates the distribution at x.
    """

    def support_size(self) -> float | int: ...

    def sample(self, k: int) -> list[T]: ...

    def evaluate(self, x: T | list[T]) -> float | list[float]: ...


def uniform_mass(categories: set[T]) -> DensityOrMass[T]:
    """Creates a uniform mass distribution over the given categories."""
    return Mass({cat: 1 / len(categories) for cat in categories})


def mass(dist: dict[T, float]) -> DensityOrMass[T]:
    """Creates a mass distribution from the given distribution."""
    return Mass(dist)


def uniform_density(loc: float = 0, scale: float = 1) -> DensityOrMass[float]:
    """Creates a uniform density distribution over the given interval.

    Args:
        loc: The lower bound of the interval.
        scale: The length of the interval
    """
    return Density(cast(ScipyDistribution, stats.uniform(loc=loc, scale=scale)))


def normal_density(loc: float = 0, scale: float = 1) -> DensityOrMass[float]:
    """Creates a normal density distribution with the given mean and standard deviation.

    Args:
        loc: The mean of the distribution.
        scale: The standard deviation of the distribution.
    """
    return Density(cast(ScipyDistribution, stats.norm(loc=loc, scale=scale)))
