"""
Categorical Distribution Sampling
=================================

This module implements sampling from categorical distributions.

Categorical distributions represent discrete outcomes with associated
probabilities. Each category is represented by an integer ID to enable
efficient manipulation and comparison within tensor operations.

Users should import from the top-level statsx package rather than
importing from this module directly.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import AbstractSet, Mapping, Sequence, cast

import random

from . import model


CategoryID = int
"""Type alias for category IDs, represented as integers."""


class CategoricalSampler:
    """Samples from a categorical distribution using integer category IDs.

    Each category is represented by an integer ID and has an associated probability.
    The sampler generates samples with assigned weights for use in ensembles.

    Args:
        distribution: A mapping from category IDs to their probabilities (must sum to 1)
            or a set of category IDs (uniform distribution).
    """

    def __init__(
        self,
        distribution: Mapping[CategoryID, model.Probability] | AbstractSet[CategoryID],
    ) -> None:
        # Ensure that we have at least one value in this distribution
        if len(distribution) <= 0:
            raise ValueError("distribution must be non-empty")

        # Handle the case of probability being uniform
        if isinstance(distribution, AbstractSet):
            probability = 1 / len(distribution)
            distribution = cast(
                Mapping[CategoryID, model.Probability],
                {cat: probability for cat in distribution},
            )

        # Validate that probabilities sum approximately to 1
        total = sum(distribution.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"probabilities must sum to 1, got {total}")

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
        subset: Sequence[CategoryID] | None = None,
        force_sample: bool = False,
    ) -> Sequence[model.Sample[CategoryID]]:
        """Sample values from this categorical distribution.

        This method uses two different sampling approaches depending on the context:

        1. Random Variates Sampling (when count < support size or force_sample=True):
            - Draws random samples based on category probabilities
            - Each sample has equal weight (1/count)
            - Represents traditional Monte Carlo sampling

        2. PDF-Based Sampling (when sampling full support or a subset):
            - Returns each category exactly once with weight proportional to its probability
            - Ensures complete coverage of the category space
            - Useful for importance sampling or weighted ensembles

        Args:
            count: Number of samples to draw.
            subset: Optional subset of category IDs to sample from.
            force_sample: Forces random variates sampling.

        Returns:
            A list of samples with weights and values.
        """
        # Sanity check: make sure that count is positive
        if count <= 0:
            raise ValueError("count must be a positive integer")

        # Check whether to sample from the full distribution or a subset
        categories = subset if subset else self.categories
        size = len(categories)

        # Sample using the random variates method if forced to do so
        # or when sampling less than the full distribution
        return (
            self.__sample_using_rvs(categories, count)
            if force_sample or count < size
            else self.__sample_using_pdf(categories, size)
        )

    def __sample_using_rvs(
        self,
        categories: Sequence[CategoryID],
        count: int,
    ) -> Sequence[model.Sample[CategoryID]]:
        """Samples from the distribution using random variates sampling.

        This approach represents traditional Monte Carlo sampling where values
        are drawn randomly according to their probability in the distribution.
        The frequency of each value in a large sample will approximate its
        probability density. Each sample receives equal weight (1/count).

        This method is most efficient when sampling fewer points than the
        support size and when independent random samples are needed.
        """

        # Each sample gets equal weight (1/count)
        sample_weight = 1 / count

        # Get probabilities for the selected categories
        probabilities = [self.distribution.get(cat, 0) for cat in categories]

        # Compute the weights to apply falling back to uniform
        # sampling when the probabilities sum to zero
        weights = probabilities if sum(probabilities) > 0 else None

        # Sample N points from the distribution
        sampled_cats = random.choices(categories, weights=weights, k=count)

        # Return samples with equal weight (1/count)
        return [
            model.Sample(weight=sample_weight, value=cat_id) for cat_id in sampled_cats
        ]

    def __sample_using_pdf(
        self,
        categories: Sequence[CategoryID],
        size: int,
    ) -> Sequence[model.Sample[CategoryID]]:
        """Samples from the distribution using the probability density function.

        This approach deterministically evaluates the probability density at each
        point in the given subset. Each value is returned exactly once with a weight
        proportional to its probability density (normalized to sum to 1).

        This method provides complete coverage of the probability space and is
        particularly useful for importance sampling, weighted ensembles, or when
        exploring the full distribution systematically.
        """
        # Compute the probability of each category
        probs = [self.distribution.get(cat, 0) for cat in categories]
        total_prob = sum(probs)

        # Fall back to uniform sampling in case of zero total probability
        if total_prob <= 0:
            return [model.Sample(weight=1 / size, value=cat) for cat in categories]

        # Return normalized probabilities
        return [
            model.Sample(weight=prob / total_prob, value=cat)
            for prob, cat in zip(probs, categories)
        ]
