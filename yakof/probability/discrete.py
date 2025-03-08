"""
Discrete Distributions
======================

This package defines the probability Mass in terms of an underlying
distribution modeled as a dict. The Mass class implements the DensityOrMass
protocol defined by the `generalized` module.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Generic, TypeVar

import random

T = TypeVar("T")
"""Type alias for the discrete distribution support."""


class Mass(Generic[T]):
    """Models a probability mass function.

    Args:
        dist: The underlying discrete distribution.

    Raises:
        ValueError: If the probabilities are not in the correct
            range (0..1) or if they do not sum to 1.
    """

    def __init__(self, dist: dict[T, float]) -> None:
        # Ensure that the probabilities are in the correct range
        if any(p < 0 or p > 1 for p in dist.values()):
            raise ValueError("probabilities must be between 0 and 1")

        # Ensure that the probabilities sum to 1
        total = sum(dist.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"probabilities must sum to 1, got {total}")

        # Save the distribution
        self.dist = dist

    def support_size(self) -> float | int:
        """Implements DensityOrMass.support_size."""
        return len(self.dist)

    def sample(self, k: int) -> list[T]:
        """Implements DensityOrMass.sample."""
        if k < 0:
            raise ValueError("k must be a zero or positive integer")
        population = list(self.dist.keys())
        weights = [self.dist[x] for x in population]
        return random.choices(
            population,
            k=k,
            weights=weights,
        )

    def evaluate(self, x: T | list[T]) -> float | list[float]:
        """Implements DensityOrMass.evaluate."""
        return (
            [self._evaluate_single(v) for v in x]
            if isinstance(x, list)
            else self._evaluate_single(x)
        )

    def _evaluate_single(self, x: T) -> float:
        return self.dist.get(x, 0)
