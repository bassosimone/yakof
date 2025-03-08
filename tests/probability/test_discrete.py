"""Tests for the yakof.probability.discrete module."""

# SPDX-License-Identifier: Apache-2.0

from typing import cast

from yakof.probability.discrete import Mass

from scipy import stats

import math
import pytest


def test_mass():
    """Ensures the mass is working as intended."""

    # === Initialization ===

    # Ensure that an empty dictionary is not allowed
    with pytest.raises(ValueError):
        Mass({})

    # Ensure that negative probabilities are not allowed
    with pytest.raises(ValueError):
        Mass({"a": -0.1, "b": 0.2, "c": 0.3, "d": 0.6})

    # Ensure that the probabilities sum to 1
    with pytest.raises(ValueError):
        Mass({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.5})

    # Create the density we want to test
    underlying = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    mass = Mass(underlying)

    # === Support Size ===

    # Ensure the support size is finite
    assert mass.support_size() == 4

    # === Sampling ===

    # Sampling random points and verifying we get something
    # that looks extracted from the given distribution.
    num_samples = 10000
    samples = mass.sample(num_samples)

    assert len(samples) == num_samples
    hist = {}
    for sample in samples:
        hist[sample] = hist.get(sample, 0) + 1
    for key in underlying:
        assert pytest.approx(hist[key] / num_samples, abs=0.01) == underlying[key]

    # Verify behavior when asked to sample zero points
    assert mass.sample(0) == []

    # Verify behavior when asked to sample negative points
    with pytest.raises(ValueError):
        mass.sample(-1)

    # === Evaluation ===

    # Ensure that we can evaluate specific points
    input_points = ["d", "b", "c", "d", "a", "b", "a"]
    expected_points = [
        0.4,
        0.2,
        0.3,
        0.4,
        0.1,
        0.2,
        0.1,
    ]
    expected_approx = pytest.approx(expected_points, abs=0.0001)
    for x, y in zip(input_points, expected_points):
        assert pytest.approx(y, abs=0.0001) == mass.evaluate(x)
    assert expected_approx == mass.evaluate(input_points)

    # Check behavior when asked to evaluate an empty list
    assert mass.evaluate([]) == []
