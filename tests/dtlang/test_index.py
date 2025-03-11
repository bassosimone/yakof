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
