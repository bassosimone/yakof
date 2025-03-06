"""
Categorical Distribution Sampling
=================================

This module provides implementation of samplers for categorical distributions.

It includes:
    - Sampler: For sampling from arbitrary categorical distributions
    - UniformSampler: For sampling from uniform categorical distributions

Categorical distributions represent discrete outcomes with associated probabilities.
Each category is represented by an integer ID to enable efficient manipulation
and comparison within tensor operations.

Users should import from the top-level statsx package rather than
importing this module directly.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Mapping, Sequence

import random

from . import model


class CategoricalSampler:
    """Samples from a categorical distribution using integer category IDs.

    Each category is represented by an integer ID and has an associated probability.
    The sampler generates samples with assigned weights for use in ensembles.

    Args:
        distribution: A mapping from category IDs to their probabilities (must sum to 1).
    """

    def __init__(self, distribution: Mapping[int, model.Probability]) -> None:
        # Ensure that we have at least one value in this distribution
        if len(distribution) <= 0:
            raise ValueError("distribution must be non-empty")

        # Validate that probabilities sum approximately to 1
        total = sum(distribution.values())
        if not (0.999 <= total <= 1.001):  # Allow for floating-point imprecision
            raise ValueError(f"Probabilities must sum to 1, got {total}")

        # Save the distribution and category IDs
        self.distribution = distribution
        self.categories = list(distribution.keys())

    def support_size(self) -> int:
        """Returns the number of categories in this distribution."""
        return len(self.distribution)

    def sample(
        self,
        count: int = 1,
        *,
        subset: Sequence[int] | None = None,
        force_sample: bool = False,
    ) -> Sequence[model.Sample[int]]:
        """Sample values from this categorical distribution.

        Args:
            count: Number of samples to draw.
            subset: Optional subset of category IDs to sample from.
            force_sample: Whether to force sampling even if count >= support_size.

        Returns:
            A list of samples with weights and values.
        """
        # Sanity check: make sure that count is positive
        if count <= 0:
            raise ValueError("count must be a positive integer")

        # Determine the set of values to sample from (either all values or a subset)
        cats = subset if subset is not None else self.categories
        size = len(cats)

        # If we need fewer samples than the support/subset size, or if forced to sample
        if force_sample or count < size:
            # Each sample gets equal weight (1/count)
            sample_weight = 1 / count

            # Get probabilities for the selected categories
            if subset is None:
                # Get probabilities directly from distribution
                probabilities = [self.distribution[cat] for cat in cats]
            else:
                # Get probabilities for the subset values
                probabilities = [self.distribution.get(cat, 0) for cat in cats]

            # Handle case where all probabilities are zero
            if sum(probabilities) <= 0:
                # Fall back to uniform sampling
                sampled_cats = random.choices(cats, k=count)
            else:
                # Weighted sampling according to the input probabilities
                sampled_cats = random.choices(cats, weights=probabilities, k=count)

            # Return samples with equal weight (1/count)
            return [model.Sample(weight=sample_weight, value=cat_id) for cat_id in sampled_cats]

        # Return all categories (or the complete subset)
        if subset is None:
            # Return all categories with original probabilities
            return [
                model.Sample(weight=prob, value=cat_id)
                for cat_id, prob in self.distribution.items()
            ]

        # Return the complete subset with normalized probabilities
        subset_probs = [self.distribution.get(cat, 0) for cat in cats]
        total_prob = sum(subset_probs)

        # Handle edge case of zero total probability
        if total_prob <= 0:
            # Fall back to uniform weights
            return [model.Sample(weight=1/size, value=cat) for cat in cats]

        # Return normalized probabilities
        return [
            model.Sample(weight=prob/total_prob, value=cat)
            for prob, cat in zip(subset_probs, cats)
        ]


class UniformCategoricalSampler(CategoricalSampler):
    """A categorical sampler with uniform probability distribution.

    Args:
        categories: A sequence of category IDs, all with equal probability.
    """

    def __init__(
        self,
        categories: Sequence[int],
    ) -> None:
        # Ensure that we have at least one value in this distribution
        if len(categories) <= 0:
            raise ValueError("categories must be non-empty")

        # Create uniform distribution and initialize parent
        probability = 1 / len(categories)
        distribution = {cat: probability for cat in categories}
        super().__init__(distribution)
