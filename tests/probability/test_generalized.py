"""Tests for the yakof.probability.generalized module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.probability.discrete import Mass

from yakof import probability
from typing import cast

import math
import pytest


def test_generalized_mass():
    """Check whether the generalized mass is usable."""
    dd = probability.mass({"a": 0.1, "b": 0.5, "c": 0.4})

    assert dd.support_size() == 3

    num_samples = 1000
    values = dd.sample(num_samples)
    assert len(values) == num_samples

    assert dd.evaluate(["a", "b", "c"]) == [0.1, 0.5, 0.4]


def test_generalized_uniform_mass():
    """Check whether the generalized uniform mass is usable."""
    dd = probability.uniform_mass({"a", "b", "c"})

    assert dd.support_size() == 3

    num_samples = 1000
    values = dd.sample(num_samples)
    assert len(values) == num_samples

    assert dd.evaluate(["a", "b", "c"]) == [1 / 3, 1 / 3, 1 / 3]


def test_generalized_discrete_constructors_equivalence():
    """Ensures that the discrete constructors are functionally equivalent."""

    # Make sure the constructor creates the correct equivalent density
    dud = probability.uniform_mass({"a", "b"})
    assert isinstance(dud, Mass)
    dud = cast(Mass, dud)
    dd = probability.mass({"a": 0.5, "b": 0.5})
    assert isinstance(dd, Mass)
    dd = cast(Mass, dd)
    assert dud.dist == dd.dist

    # Make sure the empty constructor throws
    with pytest.raises(ValueError):
        probability.uniform_mass(set())


def test_generalized_uniform_density():
    """Check whether the generalized uniform density is usable."""
    dd = probability.uniform_density(loc=0, scale=1)

    assert dd.support_size() == math.inf

    num_samples = 1000
    values = dd.sample(num_samples)
    assert len(values) == num_samples

    assert dd.evaluate([0, 0.5, 1]) == [1, 1, 1]


def test_generalized_normal_density():
    """Check whether the generalized normal density is usable."""
    dd = probability.normal_density(loc=0, scale=1)

    assert dd.support_size() == math.inf

    num_samples = 1000
    values = dd.sample(num_samples)
    assert len(values) == num_samples

    expect = pytest.approx([0.24, 0.4, 0.24], abs=0.1)
    assert dd.evaluate([-1, 0, 1]) == expect
