"""Tests for the yakof.abtesting.model5 package."""

# SPDX-License-Identifier: Apache-2.0

from yakof.abtesting.model5 import a, b

import numpy as np


def test_abtesting():
    # TODO(bassosimone): force using the same random seed. This is not
    # an issue for this test case but it's an issue in general.

    # Run using the original dt_model implementation
    rv_a = a.run()

    # Run using the yakof based implementation
    rv_b = b.run()

    # Ensure the result is the same
    assert np.allclose(rv_a, rv_b)
