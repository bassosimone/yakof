"""
Continuous Distribution Sampling
================================

This module implements sampling from continuous distributions.

Continuous distributions represent outcomes with infinite possible
values. This module is designed to work seamlessly with scipy.stats
distributions and other compatible implementations.

Users should import from the top-level statsx package rather than
importing from this module directly.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import random

from . import model
from . import scipy


class ContinuousSampler:
    """Samples from a continuous probability distribution.

    The sampler generates samples with assigned weights for use in ensembles.

    Args:
        dist: A distribution implementing the ScipyRVContinuous protocol.
    """

    def __init__(self, dist: scipy.ScipyRVContinuous) -> None:
        self.dist = dist

    def support_size(self) -> None:
        """Returns None as continuous distributions have infinite support."""
        return None

    def sample(
        self,
        count: int = 1,
        *,
        subset: Sequence[float] | None = None,
        force_sample: bool = False,
    ) -> Sequence[model.Sample[float]]:
        """Sample values from this continuous distribution.

        This method uses two different sampling approaches depending on the context:

        1. Random Variates Sampling (default or when force_sample=True):
           - Generates random samples using the distribution's random variate function
           - Each sample has equal weight (1/count)
           - For continuous distributions, this is typically the preferred method
             unless specific points need to be evaluated

        2. PDF-Based Sampling (when a subset is provided and count >= len(subset)):
           - Evaluates the probability density at each point in the subset
           - Returns each value with weight proportional to its probability density
           - Useful when you need to evaluate specific points (e.g., grid points)
             with appropriate weights

        Args:
            count: Number of samples to draw.
            subset: Optional subset of values to sample from.
            force_sample: Forces random variates sampling.

        Returns:
            A list of samples with weights and values.
        """
        # Sanity check: make sure that count is positive
        if count <= 0:
            raise ValueError("count must be a positive integer")

        return (
            self.__sample_using_rvs(count)
            if force_sample or subset is None or count < len(subset)
            else self.__sample_using_pdf(count, subset)
        )

    def __sample_using_rvs(self, count: int) -> Sequence[model.Sample[float]]:
        """Samples from the distribution using random variates sampling."""

        # Generate random samples from the distribution
        samples = self.dist.rvs(size=count)

        # Handle case of single sample
        if isinstance(samples, (float, int)):
            samples = [samples]

        # Each sample gets equal weight
        sample_weight = 1 / count
        return [model.Sample(weight=sample_weight, value=float(val)) for val in samples]

    def __sample_using_pdf(
        self,
        count: int,
        subset: Sequence[float],
    ) -> Sequence[model.Sample[float]]:
        """Samples from the distribution using the probability density function."""

        # Compute the probability of each value in the subset
        prob = self.dist.pdf(np.asarray(subset))

        # Handle case of single sample
        if isinstance(prob, (float, int)):
            prob = [prob]

        # Normalize and return the probabilities
        total_prob = np.sum(prob)
        return [
            model.Sample(weight=p / total_prob, value=v) for p, v in zip(prob, subset)
        ]
