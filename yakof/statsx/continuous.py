"""
Continuous Distribution Sampling
================================

This module provides implementations for sampling from continuous probability distributions.

It includes:
    - Distribution: A protocol compatible with scipy.stats distributions
    - Sampler: For sampling from continuous distributions

Continuous distributions represent outcomes with infinite possible values.
This module is designed to work seamlessly with scipy.stats distributions
as well as any other implementation of the Distribution protocol.

Users should import from the top-level statsx package rather than
importing this module directly.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import random

from . import model
from . import scipy


class ContinuousSampler:
    """Samples from a continuous probability distribution.

    Args:
        dist: A distribution object implementing the ContinuousDistribution protocol.
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

        Args:
            count: Number of samples to draw.
            subset: Optional subset of values to sample from.
            force_sample: Whether to force sampling even if count >= support_size.

        Returns:
            A list of samples with weights and values.
        """
        # Sanity check: make sure that count is positive
        if count <= 0:
            raise ValueError("count must be a positive integer")

        # Case 1: No subset provided - always generate random samples
        if subset is None:
            # Generate random samples from the distribution
            samples = self.dist.rvs(size=count)
            if isinstance(samples, (float, int)):  # Handle case of single sample
                samples = [samples]

            # Each sample gets equal weight
            sample_weight = 1 / count
            return [
                model.Sample(weight=sample_weight, value=float(val))
                for val in samples
            ]

        # Case 2: Subset provided and count >= subset size and not forced sampling
        # In this case, return all subset values with normalized PDF weights
        if not force_sample and count >= len(subset):
            # Calculate PDF values for each value in the subset
            pdf_values = self.dist.pdf(subset)

            # Handle the case where pdf_values is a single number
            if isinstance(pdf_values, (float, int)):
                pdf_values = [pdf_values]

            # Normalize the PDF values
            total_pdf = sum(pdf_values)
            if total_pdf <= 0:
                # Fallback to uniform weights if PDF sum is non-positive
                uniform_weight = 1.0 / len(subset)
                return [
                    model.Sample(weight=uniform_weight, value=float(val))
                    for val in subset
                ]

            # Return all subset values with normalized PDF values as weights
            return [
                model.Sample(weight=float(pdf / total_pdf), value=float(val))
                for pdf, val in zip(pdf_values, subset)
            ]

        # Case 3: Subset provided but count < subset size or force_sample=True
        # In this case, sample from the subset
        sampled_indices = random.choices(range(len(subset)), k=count)
        sampled_values = [subset[idx] for idx in sampled_indices]

        # Each sample gets equal weight
        sample_weight = 1 / count
        return [
            model.Sample(weight=sample_weight, value=float(val))
            for val in sampled_values
        ]
