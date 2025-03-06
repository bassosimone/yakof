"""
Statistical Extensions
======================

This package provides statistical extensions for working with probability distributions
and offers a unified approach to sampling from both categorical and continuous distributions
when creating weighted ensembles.

The primary goal is to provide a generalized distribution interface that works uniformly
with both categorical variables (represented as integers) and continuous variables. By
treating categories as integer values, we simplify the implementation while maintaining
flexibility - we can remap integers to their corresponding string values when needed.

Usage:
    from yakof import statsx

    # Create a categorical distribution
    cat_dist = statsx.CategoricalSampler({1: 0.3, 2: 0.7})
    samples = cat_dist.sample(count=5)

    # Create a uniform categorical distribution
    uniform_dist = statsx.UniformCategoricalSampler([1, 2, 3])

    # Working with continuous distributions
    norm_dist = statsx.ContinuousSampler(statsx.scipy_uniform(0, 1))
    samples = norm_dist.sample(count=3)

This package is designed to be imported directly as shown above, rather than
accessing its internal modules.
"""

# SPDX-License-Identifier: Apache-2.0


from .categorical import CategoricalSampler, UniformCategoricalSampler

from .continuous import ContinuousSampler

from .model import (
    Probability,
    Sample,
    Sampler,
    Value,
    Weight,
)

from .scipy import (
    ScipyRVContinuous,
    scipy_normal,
    scipy_rv_continuous_cast,
    scipy_uniform,
)

__all__ = [
    "CategoricalSampler",
    "UniformCategoricalSampler",
    "ContinuousSampler",
    "Probability",
    "Sample",
    "Sampler",
    "Value",
    "Weight",
    "ScipyRVContinuous",
    "scipy_normal",
    "scipy_rv_continuous_cast",
    "scipy_uniform",
]
