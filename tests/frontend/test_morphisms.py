"""Tests for the yakof.frontend.morphisms module."""

# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from yakof.frontend import abstract, morphisms
from yakof.numpybackend import evaluator

# Generate axes to makes tests using explicit axes more readable
x, y, z, u = morphisms.generate_canonical_axes(4)


# Define simple basis classes for testing
class Scalar:
    axes = ()


class X:
    axes = (x,)


class Y:
    axes = (y,)


class Z:
    axes = (z,)


class U:
    axes = (u,)


class XY:
    axes = (x, y)


class XZ:
    axes = (x, z)


class XU:
    axes = (x, u)


class YZ:
    axes = (y, z)


class YU:
    axes = (y, u)


class ZU:
    axes = (z, u)


class XYZU:
    axes = (x, y, z, u)


def test_generate_canonical_bases():
    assert morphisms.generate_canonical_axes(3) == (1000, 1001, 1002)
    assert morphisms.generate_canonical_axes(5) == (1000, 1001, 1002, 1003, 1004)
    assert morphisms.generate_canonical_axes(0) == ()


def test_axes_expansion_exceptions():
    # Test source not subset of destination
    with pytest.raises(ValueError, match="source must be a subset of destination"):
        morphisms.axes_expansion((z, x), (x, y))

    # Test non-monotonic source
    with pytest.raises(ValueError, match="source must have monotonic values"):
        morphisms.axes_expansion((z, x, y), (x, y, z))

    # Test non-monotonic destination
    with pytest.raises(ValueError, match="dest must have monotonic values"):
        morphisms.axes_expansion((x,), (z, x, y))


def test_axes_projection_exceptions():
    # Test destination not subset of source
    with pytest.raises(ValueError, match="destination must be a subset of source"):
        morphisms.axes_projection((x, y), (x, y, z))

    # Test non-monotonic source
    with pytest.raises(ValueError, match="source must have monotonic values"):
        morphisms.axes_projection((z, x, y), (x,))

    # Test non-monotonic destination
    with pytest.raises(ValueError, match="dest must have monotonic values"):
        morphisms.axes_projection((x, y, z), (z, x))


@pytest.mark.parametrize(
    "source,dest,expected",
    [
        # R⁰ -> R¹ expansions (scalar to vector)
        ((), (x,), 0),  # Scalar -> X
        ((), (y,), 0),  # Scalar -> Y
        ((), (z,), 0),  # Scalar -> Z
        ((), (u,), 0),  # Scalar -> U
        # R⁰ -> R² expansions (scalar to matrix)
        ((), (x, y), (0, 1)),  # Scalar -> XY
        ((), (x, z), (0, 1)),  # Scalar -> XZ
        ((), (x, u), (0, 1)),  # Scalar -> XU
        ((), (y, z), (0, 1)),  # Scalar -> YZ
        ((), (y, u), (0, 1)),  # Scalar -> YU
        ((), (z, u), (0, 1)),  # Scalar -> ZU
        # R⁰ -> R³ expansions (scalar to 3D tensor)
        ((), (x, y, z), (0, 1, 2)),  # Scalar -> XYZ
        ((), (x, y, u), (0, 1, 2)),  # Scalar -> XYU
        ((), (x, z, u), (0, 1, 2)),  # Scalar -> XZU
        ((), (y, z, u), (0, 1, 2)),  # Scalar -> YZU
        # R⁰ -> R⁴ expansions (scalar to 4D tensor)
        ((), (x, y, z, u), (0, 1, 2, 3)),  # Scalar -> XYZU
        # R¹ -> R² expansions
        (x, (x, y), 1),  # X -> XY
        (x, (x, z), 1),  # X -> XZ
        (x, (x, u), 1),  # X -> XU
        (y, (x, y), 0),  # Y -> XY
        (y, (y, z), 1),  # Y -> YZ
        (y, (y, u), 1),  # Y -> YU
        (z, (x, z), 0),  # Z -> XZ
        (z, (y, z), 0),  # Z -> YZ
        (z, (z, u), 1),  # Z -> ZU
        (u, (x, u), 0),  # U -> XU
        (u, (y, u), 0),  # U -> YU
        (u, (z, u), 0),  # U -> ZU
        # R¹ -> R³ expansions
        (x, (x, y, z), (1, 2)),  # X -> XYZ
        (x, (x, y, u), (1, 2)),  # X -> XYU
        (x, (x, z, u), (1, 2)),  # X -> XZU
        (y, (x, y, z), (0, 2)),  # Y -> XYZ
        (y, (x, y, u), (0, 2)),  # Y -> XYU
        (y, (y, z, u), (1, 2)),  # Y -> YZU
        (z, (x, z, u), (0, 2)),  # Z -> XZU
        (z, (y, z, u), (0, 2)),  # Z -> YZU
        (u, (x, y, u), (0, 1)),  # U -> XYU
        (u, (x, z, u), (0, 1)),  # U -> XZU
        (u, (y, z, u), (0, 1)),  # U -> YZU
        # R¹ -> R⁴ expansions
        (x, (x, y, z, u), (1, 2, 3)),  # X -> XYZU
        (y, (x, y, z, u), (0, 2, 3)),  # Y -> XYZU
        (z, (x, y, z, u), (0, 1, 3)),  # Z -> XYZU
        (u, (x, y, z, u), (0, 1, 2)),  # U -> XYZU
        # R² -> R³ expansions
        ((x, y), (x, y, z), 2),  # XY -> XYZ
        ((x, y), (x, y, u), 2),  # XY -> XYU
        ((x, z), (x, y, z), 1),  # XZ -> XYZ
        ((x, z), (x, z, u), 2),  # XZ -> XZU
        ((x, u), (x, y, u), 1),  # XU -> XYU
        ((x, u), (x, z, u), 1),  # XU -> XZU
        ((y, z), (x, y, z), 0),  # YZ -> XYZ
        ((y, z), (y, z, u), 2),  # YZ -> YZU
        ((y, u), (x, y, u), 0),  # YU -> XYU
        ((y, u), (y, z, u), 1),  # YU -> YZU
        ((z, u), (x, z, u), 0),  # ZU -> XZU
        ((z, u), (y, z, u), 0),  # ZU -> YZU
        # R² -> R⁴ expansions
        ((x, y), (x, y, z, u), (2, 3)),  # XY -> XYZU
        ((x, z), (x, y, z, u), (1, 3)),  # XZ -> XYZU
        ((x, u), (x, y, z, u), (1, 2)),  # XU -> XYZU
        ((y, z), (x, y, z, u), (0, 3)),  # YZ -> XYZU
        ((y, u), (x, y, z, u), (0, 2)),  # YU -> XYZU
        ((z, u), (x, y, z, u), (0, 1)),  # ZU -> XYZU
        # R³ -> R⁴ expansions
        ((x, y, z), (x, y, z, u), 3),  # XYZ -> XYZU
        ((x, y, u), (x, y, z, u), 2),  # XYU -> XYZU
        ((x, z, u), (x, y, z, u), 1),  # XZU -> XYZU
        ((y, z, u), (x, y, z, u), 0),  # YZU -> XYZU
        # Edge cases
        ((), (), ()),  # Empty -> Empty
        ((x,), (x,), ()),  # Single -> Same single
    ],
)
def test_parametrized_expansions(source, dest, expected):
    assert morphisms.axes_expansion(source, dest) == expected


@pytest.mark.parametrize(
    "source,dest,expected",
    [
        # R¹ -> R⁰ projections (vector to scalar)
        ((x,), (), 0),  # X -> Scalar
        ((y,), (), 0),  # Y -> Scalar
        ((z,), (), 0),  # Z -> Scalar
        ((u,), (), 0),  # U -> Scalar
        # R² -> R⁰ projections (matrix to scalar)
        ((x, y), (), (0, 1)),  # XY -> Scalar
        ((x, z), (), (0, 1)),  # XZ -> Scalar
        ((x, u), (), (0, 1)),  # XU -> Scalar
        ((y, z), (), (0, 1)),  # YZ -> Scalar
        ((y, u), (), (0, 1)),  # YU -> Scalar
        ((z, u), (), (0, 1)),  # ZU -> Scalar
        # R³ -> R⁰ projections (3D tensor to scalar)
        ((x, y, z), (), (0, 1, 2)),  # XYZ -> Scalar
        ((x, y, u), (), (0, 1, 2)),  # XYU -> Scalar
        ((x, z, u), (), (0, 1, 2)),  # XZU -> Scalar
        ((y, z, u), (), (0, 1, 2)),  # YZU -> Scalar
        # R⁴ -> R⁰ projections (4D tensor to scalar)
        ((x, y, z, u), (), (0, 1, 2, 3)),  # XYZU -> Scalar
        # R⁴ -> R³ projections
        ((x, y, z, u), (x, y, z), 3),  # XYZU -> XYZ
        ((x, y, z, u), (x, y, u), 2),  # XYZU -> XYU
        ((x, y, z, u), (x, z, u), 1),  # XYZU -> XZU
        ((x, y, z, u), (y, z, u), 0),  # XYZU -> YZU
        # R⁴ -> R² projections
        ((x, y, z, u), (x, y), (2, 3)),  # XYZU -> XY
        ((x, y, z, u), (x, z), (1, 3)),  # XYZU -> XZ
        ((x, y, z, u), (x, u), (1, 2)),  # XYZU -> XU
        ((x, y, z, u), (y, z), (0, 3)),  # XYZU -> YZ
        ((x, y, z, u), (y, u), (0, 2)),  # XYZU -> YU
        ((x, y, z, u), (z, u), (0, 1)),  # XYZU -> ZU
        # R⁴ -> R¹ projections
        ((x, y, z, u), x, (1, 2, 3)),  # XYZU -> X
        ((x, y, z, u), y, (0, 2, 3)),  # XYZU -> Y
        ((x, y, z, u), z, (0, 1, 3)),  # XYZU -> Z
        ((x, y, z, u), u, (0, 1, 2)),  # XYZU -> U
        # R³ -> R² projections
        ((x, y, z), (x, y), 2),  # XYZ -> XY
        ((x, y, z), (x, z), 1),  # XYZ -> XZ
        ((x, y, z), (y, z), 0),  # XYZ -> YZ
        ((x, y, u), (x, y), 2),  # XYU -> XY
        ((x, y, u), (x, u), 1),  # XYU -> XU
        ((x, y, u), (y, u), 0),  # XYU -> YU
        ((x, z, u), (x, z), 2),  # XZU -> XZ
        ((x, z, u), (x, u), 1),  # XZU -> XU
        ((x, z, u), (z, u), 0),  # XZU -> ZU
        ((y, z, u), (y, z), 2),  # YZU -> YZ
        ((y, z, u), (y, u), 1),  # YZU -> YU
        ((y, z, u), (z, u), 0),  # YZU -> ZU
        # R³ -> R¹ projections
        ((x, y, z), x, (1, 2)),  # XYZ -> X
        ((x, y, z), y, (0, 2)),  # XYZ -> Y
        ((x, y, z), z, (0, 1)),  # XYZ -> Z
        ((x, y, u), x, (1, 2)),  # XYU -> X
        ((x, y, u), y, (0, 2)),  # XYU -> Y
        ((x, y, u), u, (0, 1)),  # XYU -> U
        ((x, z, u), x, (1, 2)),  # XZU -> X
        ((x, z, u), z, (0, 2)),  # XZU -> Z
        ((x, z, u), u, (0, 1)),  # XZU -> U
        ((y, z, u), y, (1, 2)),  # YZU -> Y
        ((y, z, u), z, (0, 2)),  # YZU -> Z
        ((y, z, u), u, (0, 1)),  # YZU -> U
        # R² -> R¹ projections
        ((x, y), x, 1),  # XY -> X
        ((x, y), y, 0),  # XY -> Y
        ((x, z), x, 1),  # XZ -> X
        ((x, z), z, 0),  # XZ -> Z
        ((x, u), x, 1),  # XU -> X
        ((x, u), u, 0),  # XU -> U
        ((y, z), y, 1),  # YZ -> Y
        ((y, z), z, 0),  # YZ -> Z
        ((y, u), y, 1),  # YU -> Y
        ((y, u), u, 0),  # YU -> U
        ((z, u), z, 1),  # ZU -> Z
        ((z, u), u, 0),  # ZU -> U
        # Edge cases
        ((), (), ()),  # Empty -> Empty
        ((x,), (x,), ()),  # Single -> Same single
    ],
)
def test_parametrized_projections(source, dest, expected):
    assert morphisms.axes_projection(source, dest) == expected


def test_r4_to_r2_projections():
    # fixture1: (2,2,2,2) array reshaped from [0,1,2,...,15]*100
    fixture1 = np.arange(16).reshape(2, 2, 2, 2) * 100

    # Test all R⁴->R² projections
    for source, dest, fixture, axes in [
        (XYZU, XY, fixture1, (2, 3)),  # Sum over Z and U
        (XYZU, XZ, fixture1, (1, 3)),  # Sum over Y and U
        (XYZU, XU, fixture1, (1, 2)),  # Sum over Y and Z
        (XYZU, YZ, fixture1, (0, 3)),  # Sum over X and U
        (XYZU, YU, fixture1, (0, 2)),  # Sum over X and Z
        (XYZU, ZU, fixture1, (0, 1)),  # Sum over X and Y
    ]:
        # Create tensors using frontend
        space_source = abstract.TensorSpace(source())
        space_dest = abstract.TensorSpace(dest())
        t = space_source.placeholder("t")
        project = morphisms.ProjectUsingSum(space_source, space_dest)
        result = project(t)

        # Direct numpy projection using explicitly specified axes
        expected = np.sum(fixture, axis=axes)

        # Compare results
        actual = evaluator.evaluate(
            result.node, evaluator.StateWithoutCache({"t": fixture})
        )
        np.testing.assert_array_equal(actual, expected)


def test_r2_to_r4_expansions():
    # fixture1: (2,2) array reshaped from [0,1,2,3]
    fixture1 = np.arange(4).reshape(2, 2)

    # Test all R²->R⁴ expansions
    for source, dest, fixture, axes in [
        (XY, XYZU, fixture1, (2, 3)),  # Add Z and U dimensions
        (XZ, XYZU, fixture1, (1, 3)),  # Add Y and U dimensions
        (XU, XYZU, fixture1, (1, 2)),  # Add Y and Z dimensions
        (YZ, XYZU, fixture1, (0, 3)),  # Add X and U dimensions
        (YU, XYZU, fixture1, (0, 2)),  # Add X and Z dimensions
        (ZU, XYZU, fixture1, (0, 1)),  # Add X and Y dimensions
    ]:
        # Create tensors using frontend
        space_source = abstract.TensorSpace(source())
        space_dest = abstract.TensorSpace(dest())
        t = space_source.placeholder("t")
        expand = morphisms.ExpandDims(space_source, space_dest)
        result = expand(t)

        # Direct numpy expansion using explicitly specified axes
        expected = fixture
        for axis in axes:
            expected = np.expand_dims(expected, axis)

        # Compare results
        actual = evaluator.evaluate(
            result.node, evaluator.StateWithoutCache({"t": fixture})
        )
        np.testing.assert_array_equal(actual, expected)


def test_scalar_expansion():
    # Test expansion from scalar to 1D
    fixture = np.array(42)

    space_scalar = abstract.TensorSpace(Scalar())
    space_x = abstract.TensorSpace(X())

    t = space_scalar.placeholder("t")
    expand = morphisms.ExpandDims(space_scalar, space_x)
    result = expand(t)

    # Direct numpy expansion
    expected = np.expand_dims(fixture, 0)

    # Compare results
    actual = evaluator.evaluate(
        result.node, evaluator.StateWithoutCache({"t": fixture})
    )
    np.testing.assert_array_equal(actual, expected)


def test_scalar_projection():
    # Test projection from 1D to scalar
    fixture = np.array([1, 2, 3, 4])

    space_x = abstract.TensorSpace(X())
    space_scalar = abstract.TensorSpace(Scalar())

    t = space_x.placeholder("t")
    project = morphisms.ProjectUsingSum(space_x, space_scalar)
    result = project(t)

    # Direct numpy projection
    expected = np.sum(fixture)

    # Compare results
    actual = evaluator.evaluate(
        result.node, evaluator.StateWithoutCache({"t": fixture})
    )
    np.testing.assert_array_equal(actual, expected)
