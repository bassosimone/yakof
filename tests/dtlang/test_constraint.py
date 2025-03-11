"""Tests for the yakof.dtlang.constraint module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtlang import constraint
from yakof.dtlang import geometry

from scipy import stats


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
