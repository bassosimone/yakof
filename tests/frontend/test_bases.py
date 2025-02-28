"""Tests for the yakof.frontend.bases module."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from yakof.frontend import bases, spaces, abstract, morphisms


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


def test_space_base_correspondence():
    """Test that spaces use the correct bases."""
    assert isinstance(spaces.x.basis, bases.X)
    assert isinstance(spaces.xy.basis, bases.XY)
    assert isinstance(spaces.xyz.basis, bases.XYZ)
    assert isinstance(spaces.xyzuvw.basis, bases.XYZUVW)


def test_morphism_connections():
    """Test that morphisms connect the right spaces."""
    # Test expansion
    assert spaces.expand_x_to_xy.source == spaces.x
    assert spaces.expand_x_to_xy.dest == spaces.xy
    assert spaces.expand_xy_to_xyz.source == spaces.xy
    assert spaces.expand_xy_to_xyz.dest == spaces.xyz

    # Test projection
    assert spaces.project_xy_to_x.source == spaces.xy
    assert spaces.project_xy_to_x.dest == spaces.x
    assert spaces.project_xyz_to_xy.source == spaces.xyz
    assert spaces.project_xyz_to_xy.dest == spaces.xy
