"""Tests for the yakof.frontend.morphisms module."""

# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from yakof.frontend import abstract, morphisms


# Define the R^3 axes IDs
x, y, z = morphisms.generate_canonical_axes(3)


# Single dimension spaces
class X:
    axes = {x}


class Y:
    axes = {y}


class Z:
    axes = {z}


# Two dimensional spaces
class XY:
    axes = {x, y}


class XZ:
    axes = {x, z}


class YZ:
    axes = {y, z}


# Three dimensional space
class XYZ:
    axes = {x, y, z}


# Create tensor spaces
space_x = abstract.TensorSpace[X]()
space_y = abstract.TensorSpace[Y]()
space_z = abstract.TensorSpace[Z]()
space_xy = abstract.TensorSpace[XY]()
space_xz = abstract.TensorSpace[XZ]()
space_yz = abstract.TensorSpace[YZ]()
space_xyz = abstract.TensorSpace[XYZ]()

# R¹ -> R² expansions
expand_x_to_xy = morphisms.ExpandDims(X, XY)
expand_x_to_xz = morphisms.ExpandDims(X, XZ)
expand_y_to_xy = morphisms.ExpandDims(Y, XY)
expand_y_to_yz = morphisms.ExpandDims(Y, YZ)
expand_z_to_xz = morphisms.ExpandDims(Z, XZ)
expand_z_to_yz = morphisms.ExpandDims(Z, YZ)

# R¹ -> R³ expansions
expand_x_to_xyz = morphisms.ExpandDims(X, XYZ)
expand_y_to_xyz = morphisms.ExpandDims(Y, XYZ)
expand_z_to_xyz = morphisms.ExpandDims(Z, XYZ)

# R² -> R³ expansions
expand_xy_to_xyz = morphisms.ExpandDims(XY, XYZ)
expand_xz_to_xyz = morphisms.ExpandDims(XZ, XYZ)
expand_yz_to_xyz = morphisms.ExpandDims(YZ, XYZ)

# R³ -> R² projections
project_xyz_to_xy = morphisms.ProjectUsingSum(XYZ, XY)
project_xyz_to_xz = morphisms.ProjectUsingSum(XYZ, XZ)
project_xyz_to_yz = morphisms.ProjectUsingSum(XYZ, YZ)

# R² -> R¹ projections
project_xy_to_x = morphisms.ProjectUsingSum(XY, X)
project_xy_to_y = morphisms.ProjectUsingSum(XY, Y)
project_xz_to_x = morphisms.ProjectUsingSum(XZ, X)
project_xz_to_z = morphisms.ProjectUsingSum(XZ, Z)
project_yz_to_y = morphisms.ProjectUsingSum(YZ, Y)
project_yz_to_z = morphisms.ProjectUsingSum(YZ, Z)

# R³ -> R¹ projections
project_xyz_to_x = morphisms.ProjectUsingSum(XYZ, X)
project_xyz_to_y = morphisms.ProjectUsingSum(XYZ, Y)
project_xyz_to_z = morphisms.ProjectUsingSum(XYZ, Z)


def test_generate_canonical_bases():
    assert morphisms.generate_canonical_axes(3) == (0, 1, 2)
    assert morphisms.generate_canonical_axes(5) == (0, 1, 2, 3, 4)
    assert morphisms.generate_canonical_axes(0) == ()


# TODO(bassosimone): think about corner cases


def test_axes_expansion():
    # Single dimension expansions
    assert morphisms.axes_expansion(z, (y, z)) == 0
    assert morphisms.axes_expansion((x, z), (x, y, z)) == 1

    # Multiple dimension expansions
    assert morphisms.axes_expansion(z, (x, y, z)) == (0, 1)

    # Empty expansion (should this be allowed?)
    assert morphisms.axes_expansion((0,), (0,)) == ()


def test_axes_projection():
    # Single dimension projections
    assert morphisms.axes_projection((x, y, z), (x, z)) == 1

    # Multiple dimension projections
    assert morphisms.axes_projection((x, y, z), z) == (0, 1)


def test_expand_dims_morphism():
    x, y = space_x.placeholder("x"), space_y.placeholder("y")
    pass


def test_project_using_sum_morphism():
    project = morphisms.ProjectUsingSum(XYZ, XZ)
    # Need to create a tensor and verify projection...


def test_invalid_projections():
    # Test error cases
    with pytest.raises(ValueError):
        morphisms.axes_projection((0,), (0, 1))  # Can't project to larger space


def test_composite_transformations():
    # Test that composing expansion and projection works as expected
    # e.g., Z -> XYZ -> XZ should give expected result
    pass


@pytest.mark.parametrize(
    "source,dest,expected",
    [
        # R¹ -> R² expansions
        (z, (y, z), 0),  # Z -> YZ
        (z, (x, z), 0),  # Z -> XZ
        (y, (x, y), 0),  # Y -> XY
        (y, (y, z), 1),  # Y -> YZ
        (x, (x, y), 1),  # X -> XY
        (x, (x, z), 1),  # X -> XZ
        # R¹ -> R³ expansions
        (x, (x, y, z), (1, 2)),  # X -> XYZ
        (y, (x, y, z), (0, 2)),  # Y -> XYZ
        (z, (x, y, z), (0, 1)),  # Z -> XYZ
        # R² -> R³ expansions
        ((x, y), (x, y, z), 2),  # XY -> XYZ
        ((x, z), (x, y, z), 1),  # XZ -> XYZ
        ((y, z), (x, y, z), 0),  # YZ -> XYZ
        # Edge cases
        ((), (), ()),  # Empty -> Empty
        ((x,), (x,), ()),  # Single -> Same single
        ((x,), (x, y, z), (1, 2)),  # Single -> Full space
    ],
)
def test_parametrized_expansions(source, dest, expected):
    assert morphisms.axes_expansion(source, dest) == expected


@pytest.mark.parametrize(
    "source,dest,expected",
    [
        # R³ -> R² projections
        ((x, y, z), (x, y), 2),  # XYZ -> XY
        ((x, y, z), (x, z), 1),  # XYZ -> XZ
        ((x, y, z), (y, z), 0),  # XYZ -> YZ
        # R² -> R¹ projections
        ((x, y), x, 1),  # XY -> X
        ((x, y), y, 0),  # XY -> Y
        ((x, z), x, 1),  # XZ -> X
        ((x, z), z, 0),  # XZ -> Z
        ((y, z), y, 1),  # YZ -> Y
        ((y, z), z, 0),  # YZ -> Z
        # R³ -> R¹ projections
        ((x, y, z), x, (1, 2)),  # XYZ -> X
        ((x, y, z), y, (0, 2)),  # XYZ -> Y
        ((x, y, z), z, (0, 1)),  # XYZ -> Z
        # Edge cases
        ((), (), ()),  # Empty -> Empty
        ((x,), (x,), ()),  # Single -> Same single
        ((x, y, z), x, (1, 2)),  # Full space -> Single
    ],
)
def test_parametrized_projections(source, dest, expected):
    assert morphisms.axes_projection(source, dest) == expected
