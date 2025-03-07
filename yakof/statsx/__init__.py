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

This package is designed to be imported directly as shown above,
rather than accessing its internal modules.

Sampling Strategy
-----------------

This package implements two complementary sampling approaches:

1. Random Variates Sampling (RVS method):
   - Draws random samples with probability matching the distribution
   - Each sample receives equal weight (1/count)
   - Used when sampling fewer points than the distribution's support size

2. Probability Density Function Sampling (PDF method):
   - Returns values with weights proportional to their probability density
   - Used when sampling the entire support or a specified subset
   - Provides complete coverage of the probability space

The package automatically selects the appropriate method based on the sampling
requirements, but you can force RVS using the force_sample parameter.

Usage
-----

    from yakof import statsx

    # Create a categorical distribution
    cat_dist = statsx.CategoricalSampler({1: 0.3, 2: 0.7})
    samples = cat_dist.sample(count=5)

    # Same as above, with uniform distribution
    ucat_dist = statsx.UniformCategoricalSampler([1, 2])
    samples = ucat_dist.sample(count=5)

    # Working with continuous distributions
    norm_dist = statsx.ContinuousSampler(statsx.scipy_uniform(0, 1))
    samples = norm_dist.sample(count=5)

    # Force RVS sampling even when sampling the full support
    samples = cat_dist.sample(count=10, force_sample=True)
"""

# SPDX-License-Identifier: Apache-2.0


from .categorical import CategoricalSampler

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
