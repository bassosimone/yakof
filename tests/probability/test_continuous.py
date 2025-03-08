"""Tests for the yakof.probability.continuous module."""

# SPDX-License-Identifier: Apache-2.0

from typing import cast

from yakof.probability.continuous import Density, ScipyDistribution

from scipy import stats

import math
import pytest


def test_density():
    """Ensures the density is working as intended."""

    # === Initialization ===

    # Create the density we want to test
    underlying = stats.norm()
    density = Density(cast(ScipyDistribution, underlying))

    # === Support Size ===

    # Ensure the support size is infinite
    assert density.support_size() == math.inf

    # === Sampling ===

    # Sampling random points and verifying we get something
    # that looks extracted from a normal distribution.
    #
    # With 10k samples, the standard error should be around 0.01
    # so here we're allowing for four standard errors.
    #
    # Also, the standard error for the standard deviation should
    # be around 0.007, so we are giving plenty of variation.
    num_samples = 10000
    samples = density.sample(num_samples)

    assert len(samples) == num_samples
    avg = sum(samples) / num_samples
    assert avg == pytest.approx(0, abs=0.04)

    std_dev = math.sqrt(sum((sample - avg) ** 2 for sample in samples) / num_samples)
    assert std_dev == pytest.approx(1, abs=0.021)

    # Verify behavior when asked to sample zero points
    assert density.sample(0) == []

    # Verify behavior when asked to sample negative points
    with pytest.raises(ValueError):
        density.sample(-1)

    # === Evaluation ===

    # Ensure that we can evaluate specific points
    input_points = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    expected_points = [
        0.00443,
        0.05399,
        0.24197,
        0.39894,
        0.24197,
        0.05399,
        0.00443,
    ]
    expected_approx = pytest.approx(expected_points, abs=0.0001)
    for x, y in zip(input_points, expected_points):
        assert pytest.approx(y, abs=0.0001) == density.evaluate(x)
    assert expected_approx == density.evaluate(input_points)

    # Check behavior when asked to evaluate an empty list
    assert density.evaluate([]) == []
