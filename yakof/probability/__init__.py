"""
Generalized Probability Distributions
=====================================

This package provides a unified interface for working with probability
distributions, abstracting over continuous and discrete ones.

Namely, a generalized probability distribution is modeled by the
`DensityOrMass` protocol, which defines three methods:

    support_size: Returns the size of the support.
    sample: Generates a sample of size k.
    evaluate: Evaluates the distribution at x.

Additionally, this package provides the factory functions for
creating several continuous and discrete distributions.
"""

# SPDX-License-Identifier: Apache-2.0

from .generalized import (
    DensityOrMass,
    normal_density,
    uniform_density,
    mass,
    uniform_mass,
)

__all__ = [
    "DensityOrMass",
    "normal_density",
    "uniform_density",
    "mass",
    "uniform_mass",
]
