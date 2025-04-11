"""Tests for the yakof.frontend.spaces module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.frontend import bases, spaces


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
        spaces.x,
        spaces.y,
        spaces.z,
        spaces.u,
        spaces.v,
        spaces.w,
        # Sample from R²
        spaces.xy,
        spaces.xz,
        spaces.yz,
        spaces.uv,
        # Sample from R³
        spaces.xyz,
        spaces.xyu,
        spaces.zuv,
        # Sample from R⁴+
        spaces.xyzu,
        spaces.xyzuv,
        spaces.xyzuvw,
    ]

    for space in all_spaces:
        axes = space.basis.axes
        for i in range(1, len(axes)):
            assert axes[i] > axes[i - 1], f"Axes not monotonic in {space.basis.__class__.__name__}"


def test_expansion_morphisms():
    """Test that expansion morphisms connect the right spaces."""
    # Test R¹ -> R² expansions
    assert spaces.expand_x_to_xy.source == spaces.x
    assert spaces.expand_x_to_xy.dest == spaces.xy
    assert spaces.expand_y_to_yz.source == spaces.y
    assert spaces.expand_y_to_yz.dest == spaces.yz

    # Test R² -> R³ expansions
    assert spaces.expand_xy_to_xyz.source == spaces.xy
    assert spaces.expand_xy_to_xyz.dest == spaces.xyz
    assert spaces.expand_uv_to_zuv.source == spaces.uv
    assert spaces.expand_uv_to_zuv.dest == spaces.zuv

    # Test higher dimension expansions
    assert spaces.expand_xyzuv_to_xyzuvw.source == spaces.xyzuv
    assert spaces.expand_xyzuv_to_xyzuvw.dest == spaces.xyzuvw


def test_projection_morphisms():
    """Test that projection morphisms connect the right spaces."""
    # Test R² -> R¹ projections
    assert spaces.project_xy_to_x_using_sum.source == spaces.xy
    assert spaces.project_xy_to_x_using_sum.dest == spaces.x
    assert spaces.project_yz_to_z_using_sum.source == spaces.yz
    assert spaces.project_yz_to_z_using_sum.dest == spaces.z

    # Test R³ -> R² projections
    assert spaces.project_xyz_to_xy_using_sum.source == spaces.xyz
    assert spaces.project_xyz_to_xy_using_sum.dest == spaces.xy
    assert spaces.project_uvw_to_vw_using_sum.source == spaces.uvw
    assert spaces.project_uvw_to_vw_using_sum.dest == spaces.vw

    # Test higher dimension projections
    assert spaces.project_xyzuvw_to_xyzuv_using_sum.source == spaces.xyzuvw
    assert spaces.project_xyzuvw_to_xyzuv_using_sum.dest == spaces.xyzuv
