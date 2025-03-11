"""Tests for the yakof.dtlang.constraint module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtlang import constraint
from yakof.dtlang import geometry

from scipy import stats

import numpy as np


def test_constraint_assign_tensors():
    """Make sure that we can safely assign tensors to the constraint."""

    a = geometry.space.placeholder("a")
    b = geometry.space.placeholder("b")
    c = constraint.Constraint(a, b)
    assert c.usage == a
    assert c.capacity == b


def test_constraint_assign_distribution():
    """Make sure that we can assign a distribution capacity."""
    a = geometry.space.placeholder("a")
    b = stats.uniform()
    c = constraint.Constraint(a, b)
    assert c.usage == a
    assert c.capacity == b


def test_constraint_with_name():
    """Test constraint initialization with a name."""
    a = geometry.space.placeholder("a")
    b = geometry.space.placeholder("b")
    c = constraint.Constraint(a, b, name="test_constraint")

    assert c.name == "test_constraint"
    assert c.usage == a
    assert c.capacity == b


def test_constraint_protocol():
    """Test the CumulativeDistribution protocol compatibility."""

    class TestDistribution:
        def cdf(self, x, *args, **kwds):
            return 0.5 if isinstance(x, float) else np.ones_like(x) * 0.5

    # Check if our test class is recognized as a CumulativeDistribution
    test_dist = TestDistribution()
    assert isinstance(test_dist, constraint.CumulativeDistribution)

    # Use it in a constraint
    a = geometry.space.placeholder("a")
    c = constraint.Constraint(a, test_dist)
    assert c.capacity is test_dist
