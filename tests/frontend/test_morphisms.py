"""Tests for the yakof.frontend.morphisms module."""

# SPDX-License-Identifier: Apache-2.0

import pytest
from yakof.frontend import bases, morphisms, spaces


def test_generate_canonical_bases():
    assert morphisms.generate_canonical_axes(3) == (1000, 1001, 1002)
    assert morphisms.generate_canonical_axes(5) == (1000, 1001, 1002, 1003, 1004)
    assert morphisms.generate_canonical_axes(0) == ()


# Generate axes to makes tests using explicit axes more readable
x, y, z, u = morphisms.generate_canonical_axes(4)


# TODO(bassosimone): think about corner cases


def test_axes_expansion():
    # Single dimension expansions
    assert morphisms.axes_expansion(2, (1, 2)) == 0
    assert morphisms.axes_expansion((0, 2), (0, 1, 2)) == 1

    # Multiple dimension expansions
    assert morphisms.axes_expansion(2, (0, 1, 2)) == (0, 1)

    # Empty expansion (should this be allowed?)
    assert morphisms.axes_expansion((0,), (0,)) == ()


def test_axes_projection():
    # Single dimension projections
    assert morphisms.axes_projection((0, 1, 2), (0, 2)) == 1

    # Multiple dimension projections
    assert morphisms.axes_projection((0, 1, 2), 2) == (0, 1)


def test_expand_dims_morphism():
    x, y = spaces.x.placeholder("x"), spaces.y.placeholder("y")
    pass


def test_project_using_sum_morphism():
    project = morphisms.ProjectUsingSum(bases.XYZ, bases.XZ)
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
        (0, (0, 1), 1),  # X -> XY
        (0, (0, 2), 1),  # X -> XZ
        (0, (0, 3), 1),  # X -> XU
        (1, (0, 1), 0),  # Y -> XY
        (1, (1, 2), 1),  # Y -> YZ
        (1, (1, 3), 1),  # Y -> YU
        (2, (0, 2), 0),  # Z -> XZ
        (2, (1, 2), 0),  # Z -> YZ
        (2, (2, 3), 1),  # Z -> ZU
        (3, (0, 3), 0),  # U -> XU
        (3, (1, 3), 0),  # U -> YU
        (3, (2, 3), 0),  # U -> ZU
        # R¹ -> R³ expansions
        (0, (0, 1, 2), (1, 2)),  # X -> XYZ
        (0, (0, 1, 3), (1, 2)),  # X -> XYU
        (0, (0, 2, 3), (1, 2)),  # X -> XZU
        (1, (0, 1, 2), (0, 2)),  # Y -> XYZ
        (1, (0, 1, 3), (0, 2)),  # Y -> XYU
        (1, (1, 2, 3), (1, 2)),  # Y -> YZU
        (2, (0, 2, 3), (0, 2)),  # Z -> XZU
        (2, (1, 2, 3), (0, 2)),  # Z -> YZU
        (3, (0, 1, 3), (0, 1)),  # U -> XYU
        (3, (0, 2, 3), (0, 1)),  # U -> XZU
        (3, (1, 2, 3), (0, 1)),  # U -> YZU
        # R¹ -> R⁴ expansions
        (0, (0, 1, 2, 3), (1, 2, 3)),  # X -> XYZU
        (1, (0, 1, 2, 3), (0, 2, 3)),  # Y -> XYZU
        (2, (0, 1, 2, 3), (0, 1, 3)),  # Z -> XYZU
        (3, (0, 1, 2, 3), (0, 1, 2)),  # U -> XYZU
        # R² -> R³ expansions
        ((0, 1), (0, 1, 2), 2),  # XY -> XYZ
        ((0, 1), (0, 1, 3), 2),  # XY -> XYU
        ((0, 2), (0, 1, 2), 1),  # XZ -> XYZ
        ((0, 2), (0, 2, 3), 2),  # XZ -> XZU
        ((0, 3), (0, 1, 3), 1),  # XU -> XYU
        ((0, 3), (0, 2, 3), 1),  # XU -> XZU
        ((1, 2), (0, 1, 2), 0),  # YZ -> XYZ
        ((1, 2), (1, 2, 3), 2),  # YZ -> YZU
        ((1, 3), (0, 1, 3), 0),  # YU -> XYU
        ((1, 3), (1, 2, 3), 1),  # YU -> YZU
        ((2, 3), (0, 2, 3), 0),  # ZU -> XZU
        ((2, 3), (1, 2, 3), 0),  # ZU -> YZU
        # R² -> R⁴ expansions
        ((0, 1), (0, 1, 2, 3), (2, 3)),  # XY -> XYZU
        ((0, 2), (0, 1, 2, 3), (1, 3)),  # XZ -> XYZU
        ((0, 3), (0, 1, 2, 3), (1, 2)),  # XU -> XYZU
        ((1, 2), (0, 1, 2, 3), (0, 3)),  # YZ -> XYZU
        ((1, 3), (0, 1, 2, 3), (0, 2)),  # YU -> XYZU
        ((2, 3), (0, 1, 2, 3), (0, 1)),  # ZU -> XYZU
        # R³ -> R⁴ expansions
        ((0, 1, 2), (0, 1, 2, 3), 3),  # XYZ -> XYZU
        ((0, 1, 3), (0, 1, 2, 3), 2),  # XYU -> XYZU
        ((0, 2, 3), (0, 1, 2, 3), 1),  # XZU -> XYZU
        ((1, 2, 3), (0, 1, 2, 3), 0),  # YZU -> XYZU
        # Edge cases
        ((), (), ()),  # Empty -> Empty
        ((0,), (0,), ()),  # Single -> Same single
        ((0,), (0, 1, 2, 3), (1, 2, 3)),  # Single -> Full R⁴ space
    ],
)
def test_parametrized_expansions(source, dest, expected):
    assert morphisms.axes_expansion(source, dest) == expected


@pytest.mark.parametrize(
    "source,dest,expected",
    [
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
