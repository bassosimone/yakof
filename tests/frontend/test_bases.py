"""Tests for the yakof.frontend.bases module."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from yakof.frontend import bases


def test_base_dimensions():
    """Test that bases have correct dimensions and monotonic axes."""
    # Sample from each dimension
    r1_bases = [bases.X, bases.Y, bases.Z]
    r2_bases = [bases.XY, bases.YZ, bases.ZU]
    r3_bases = [bases.XYZ, bases.YZU, bases.ZUV]

    for base_cls in r1_bases:
        assert len(base_cls().axes) == 1
    for base_cls in r2_bases:
        assert len(base_cls().axes) == 2
    for base_cls in r3_bases:
        assert len(base_cls().axes) == 3

    # Test monotonic property with a sample
    for base_cls in [bases.XY, bases.XYZ, bases.XYZU, bases.XYZUVW]:
        axes = base_cls().axes
        assert all(axes[i] < axes[i + 1] for i in range(len(axes) - 1))
