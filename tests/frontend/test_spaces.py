"""Tests for the yakof.frontend.spaces module."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from yakof.frontend import bases, spaces, abstract, morphisms

def test_canonical_spaces_dimensions():
    """Test that spaces of each dimension have the correct bases."""
    # R¹ spaces (single axis)
    r1_spaces = [spaces.x, spaces.y, spaces.z, spaces.u, spaces.v, spaces.w]
    r1_bases = [bases.X, bases.Y, bases.Z, bases.U, bases.V, bases.W]

    for space, base_class in zip(r1_spaces, r1_bases):
        assert isinstance(space.basis, base_class)
        assert len(space.basis.axes) == 1

    # Test representative R² spaces
    assert isinstance(spaces.xy.basis, bases.XY)
    assert isinstance(spaces.yz.basis, bases.YZ)
    assert isinstance(spaces.uv.basis, bases.UV)
    assert len(spaces.xy.basis.axes) == 2

    # Test representative R³ spaces
    assert isinstance(spaces.xyz.basis, bases.XYZ)
    assert isinstance(spaces.yuv.basis, bases.YUV)
    assert len(spaces.xyz.basis.axes) == 3

    # Test representative higher-dimensional spaces
    assert isinstance(spaces.xyzu.basis, bases.XYZU)
    assert isinstance(spaces.xyzuv.basis, bases.XYZUV)
    assert isinstance(spaces.xyzuvw.basis, bases.XYZUVW)
    assert len(spaces.xyzuvw.basis.axes) == 6

def test_monotonic_axes_in_spaces():
    """Test that axes in each space's basis are monotonically increasing."""
    all_spaces = [
        # R¹
        spaces.x, spaces.y, spaces.z, spaces.u, spaces.v, spaces.w,
        # Sample from R²
        spaces.xy, spaces.xz, spaces.yz, spaces.uv,
        # Sample from R³
        spaces.xyz, spaces.xyu, spaces.zuv,
        # Sample from R⁴+
        spaces.xyzu, spaces.xyzuv, spaces.xyzuvw
    ]

    for space in all_spaces:
        axes = space.basis.axes
        for i in range(1, len(axes)):
            assert axes[i] > axes[i-1], f"Axes not monotonic in {space.basis.__class__.__name__}"

def test_space_types():
    """Test that all spaces are TensorSpace instances."""
    all_spaces = [
        # R¹
        spaces.x, spaces.y, spaces.z, spaces.u, spaces.v, spaces.w,
        # Sample from R²
        spaces.xy, spaces.yz, spaces.vw,
        # Sample from R³
        spaces.xyz, spaces.yuv, spaces.uvw,
        # Sample from higher dimensions
        spaces.xyzu, spaces.xyzuv, spaces.xyzuvw
    ]

    for space in all_spaces:
        assert isinstance(space, abstract.TensorSpace)

def test_expansion_morphisms():
    """Test that expansion morphisms connect the right spaces."""
    # Test R¹ -> R² expansions
    assert spaces.expand_x_to_xy.source == spaces.x
    assert spaces.expand_x_to_xy.target == spaces.xy
    assert spaces.expand_y_to_yz.source == spaces.y
    assert spaces.expand_y_to_yz.target == spaces.yz

    # Test R² -> R³ expansions
    assert spaces.expand_xy_to_xyz.source == spaces.xy
    assert spaces.expand_xy_to_xyz.target == spaces.xyz
    assert spaces.expand_uv_to_zuv.source == spaces.uv
    assert spaces.expand_uv_to_zuv.target == spaces.zuv

    # Test higher dimension expansions
    assert spaces.expand_xyzuv_to_xyzuvw.source == spaces.xyzuv
    assert spaces.expand_xyzuv_to_xyzuvw.target == spaces.xyzuvw

def test_projection_morphisms():
    """Test that projection morphisms connect the right spaces."""
    # Test R² -> R¹ projections
    assert spaces.project_xy_to_x.source == spaces.xy
    assert spaces.project_xy_to_x.target == spaces.x
    assert spaces.project_yz_to_z.source == spaces.yz
    assert spaces.project_yz_to_z.target == spaces.z

    # Test R³ -> R² projections
    assert spaces.project_xyz_to_xy.source == spaces.xyz
    assert spaces.project_xyz_to_xy.target == spaces.xy
    assert spaces.project_uvw_to_vw.source == spaces.uvw
    assert spaces.project_uvw_to_vw.target == spaces.vw

    # Test higher dimension projections
    assert spaces.project_xyzuvw_to_xyzuv.source == spaces.xyzuvw
    assert spaces.project_xyzuvw_to_xyzuv.target == spaces.xyzuv

def test_count_spaces_and_morphisms():
    """Test that we have the expected number of spaces and morphisms."""
    # Count spaces by dimension
    r1_spaces = [attr for attr in dir(spaces) if attr in "xyzuvw"]
    assert len(r1_spaces) == 6  # x, y, z, u, v, w

    r2_spaces = [attr for attr in dir(spaces)
                if len(attr) == 2 and all(c in "xyzuvw" for c in attr)]
    assert len(r2_spaces) == 15  # C(6,2) = 15 combinations

    # Count expansion morphisms
    r1_to_r2_expands = [attr for attr in dir(spaces)
                      if attr.startswith("expand_") and len(attr) == 11]
    assert len(r1_to_r2_expands) == 30  # Each R¹ space expands to 5 R² spaces

    # Count projection morphisms
    r2_to_r1_projects = [attr for attr in dir(spaces)
                       if attr.startswith("project_") and "_to_" in attr
                       and attr[-2] in "xyzuvw" and len(attr) == 13]
    assert len(r2_to_r1_projects) == 30  # Each R² space projects to 2 R¹ spaces

def test_morphism_types():
    """Test that morphisms are of correct types."""
    # Test expansions
    assert isinstance(spaces.expand_x_to_xy, morphisms.ExpandDims)
    assert isinstance(spaces.expand_xyz_to_xyzu, morphisms.ExpandDims)

    # Test projections
    assert isinstance(spaces.project_xy_to_x, morphisms.ProjectUsingSum)
    assert isinstance(spaces.project_xyz_to_xy, morphisms.ProjectUsingSum)
