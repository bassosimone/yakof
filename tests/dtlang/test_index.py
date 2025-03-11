"""Tests for the yakof.dtlang.index module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtlang import index

from scipy import stats

import numpy as np


def test_index_with_scalar():
    """Make sure an index initialized with a scalar works as intended."""

    idx = index.Index("idx", 0)
    assert idx.name == "idx"

    values = idx.distribution.rvs(100)
    assert isinstance(values, np.ndarray)
    assert values == 0


def test_index_with_distribution():
    """Make sure an index initialized with a distribution works as intended."""

    idx = index.Index("idx", stats.uniform())
    assert idx.name == "idx"

    values = idx.distribution.rvs(100)
    assert len(values) == 100
    assert all(0 <= v <= 1 for v in values)


def test_index_with_custom_distribution():
    """Test Index with a custom distribution class."""

    class CustomDistribution:
        def rvs(self, size=1):
            return np.ones(size) * 42

    custom_dist = CustomDistribution()
    idx = index.Index("custom", custom_dist)

    values = idx.distribution.rvs(10)
    assert len(values) == 10
    assert all(v == 42 for v in values)
