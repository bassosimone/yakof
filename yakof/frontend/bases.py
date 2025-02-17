"""
Tensor Bases and Transformations
================================

This module provides type definitions and transformations for tensor spaces
up to R³, focusing on the X, Y, and Z dimensions.

Type Definitions:
---------------
- R¹: X, Y, Z (single axes)
- R²: XY, XZ, YZ (pairs of axes)
- R³: XYZ (all three axes)

Expansion Operations:
------------------
When expanding from n to n+1 dimensions, we add a new axis using numpy/graph
conventions for physical array positions, NOT using our logical axis numbering:

1. R¹ -> R² expansions:
   - X -> XY: add Y at position 1
   - X -> XZ: add Z at position 1
   - Y -> XY: add X at position 0
   - Y -> YZ: add Z at position 1
   - Z -> XZ: add X at position 0
   - Z -> YZ: add Y at position 0

2. R² -> R³ expansions:
   - XY -> XYZ: add Z at position 2
   - XZ -> XYZ: add Y at position 1
   - YZ -> XYZ: add X at position 0

Projection Operations:
-------------------
When projecting from n to n-1 dimensions, we reduce along the axis we want
to eliminate using its physical position in the array:

1. R³ -> R² projections:
   - XYZ -> XY: reduce along axis 2 (Z)
   - XYZ -> XZ: reduce along axis 1 (Y)
   - XYZ -> YZ: reduce along axis 0 (X)

2. R² -> R¹ projections:
   - XY -> X: reduce along axis 1 (Y)
   - XY -> Y: reduce along axis 0 (X)
   - XZ -> X: reduce along axis 1 (Z)
   - XZ -> Z: reduce along axis 0 (X)
   - YZ -> Y: reduce along axis 1 (Z)
   - YZ -> Z: reduce along axis 0 (Y)

Example:
-------
    >>> from yakof.frontend import abstract, bases
    >>> space_x = abstract.TensorSpace[bases.X]()
    >>> x = space_x.placeholder("x")        # tensor in R¹
    >>> xy = bases.expand_x_to_xy(x)        # expand to R²
    >>> xyz = bases.expand_xy_to_xyz(xy)    # expand to R³
    >>> xy2 = bases.project_xyz_to_xy(xyz)  # project back to R²
    >>> x2 = bases.project_xy_to_x(xy2)     # project back to R¹
"""

from yakof.frontend import graph
from yakof.frontend.abstract import Tensor


# Single axes (R¹)
class X:
    """Type representing a tensor in R¹ along the X axis."""


class Y:
    """Type representing a tensor in R¹ along the Y axis."""


class Z:
    """Type representing a tensor in R¹ along the Z axis."""


# R² combinations
class XY:
    """Type representing a tensor in R² along the X and Y axes."""


class XZ:
    """Type representing a tensor in R² along the X and Z axes."""


class YZ:
    """Type representing a tensor in R² along the Y and Z axes."""


# R³
class XYZ:
    """Type representing a tensor in R³ along the X, Y and Z axes."""


# Expansions from R¹ to R²
def expand_x_to_xy(t: Tensor[X]) -> Tensor[XY]:
    """Expand X tensor by adding Y dimension at position 1."""
    return Tensor[XY](graph.expand_dims(t.t, axis=1))


def expand_x_to_xz(t: Tensor[X]) -> Tensor[XZ]:
    """Expand X tensor by adding Z dimension at position 1."""
    return Tensor[XZ](graph.expand_dims(t.t, axis=1))


def expand_y_to_xy(t: Tensor[Y]) -> Tensor[XY]:
    """Expand Y tensor by adding X dimension at position 0."""
    return Tensor[XY](graph.expand_dims(t.t, axis=0))


def expand_y_to_yz(t: Tensor[Y]) -> Tensor[YZ]:
    """Expand Y tensor by adding Z dimension at position 1."""
    return Tensor[YZ](graph.expand_dims(t.t, axis=1))


def expand_z_to_xz(t: Tensor[Z]) -> Tensor[XZ]:
    """Expand Z tensor by adding X dimension at position 0."""
    return Tensor[XZ](graph.expand_dims(t.t, axis=0))


def expand_z_to_yz(t: Tensor[Z]) -> Tensor[YZ]:
    """Expand Z tensor by adding Y dimension at position 0."""
    return Tensor[YZ](graph.expand_dims(t.t, axis=0))


# Expansions from R² to R³
def expand_xy_to_xyz(t: Tensor[XY]) -> Tensor[XYZ]:
    """Expand XY tensor by adding Z dimension at position 2."""
    return Tensor[XYZ](graph.expand_dims(t.t, axis=2))


def expand_xz_to_xyz(t: Tensor[XZ]) -> Tensor[XYZ]:
    """Expand XZ tensor by adding Y dimension at position 1."""
    return Tensor[XYZ](graph.expand_dims(t.t, axis=1))


def expand_yz_to_xyz(t: Tensor[YZ]) -> Tensor[XYZ]:
    """Expand YZ tensor by adding X dimension at position 0."""
    return Tensor[XYZ](graph.expand_dims(t.t, axis=0))


# Projections from R² to R¹
def project_xy_to_x(t: Tensor[XY]) -> Tensor[X]:
    """Project XY tensor to X by reducing Y dimension (axis 1)."""
    return Tensor[X](graph.reduce_mean(t.t, axis=1))


def project_xy_to_y(t: Tensor[XY]) -> Tensor[Y]:
    """Project XY tensor to Y by reducing X dimension (axis 0)."""
    return Tensor[Y](graph.reduce_mean(t.t, axis=0))


def project_xz_to_x(t: Tensor[XZ]) -> Tensor[X]:
    """Project XZ tensor to X by reducing Z dimension (axis 1)."""
    return Tensor[X](graph.reduce_mean(t.t, axis=1))


def project_xz_to_z(t: Tensor[XZ]) -> Tensor[Z]:
    """Project XZ tensor to Z by reducing X dimension (axis 0)."""
    return Tensor[Z](graph.reduce_mean(t.t, axis=0))


def project_yz_to_y(t: Tensor[YZ]) -> Tensor[Y]:
    """Project YZ tensor to Y by reducing Z dimension (axis 1)."""
    return Tensor[Y](graph.reduce_mean(t.t, axis=1))


def project_yz_to_z(t: Tensor[YZ]) -> Tensor[Z]:
    """Project YZ tensor to Z by reducing Y dimension (axis 0)."""
    return Tensor[Z](graph.reduce_mean(t.t, axis=0))


# Projections from R³ to R²
def project_xyz_to_xy(t: Tensor[XYZ]) -> Tensor[XY]:
    """Project XYZ tensor to XY by reducing Z dimension (axis 2)."""
    return Tensor[XY](graph.reduce_mean(t.t, axis=2))


def project_xyz_to_xz(t: Tensor[XYZ]) -> Tensor[XZ]:
    """Project XYZ tensor to XZ by reducing Y dimension (axis 1)."""
    return Tensor[XZ](graph.reduce_mean(t.t, axis=1))


def project_xyz_to_yz(t: Tensor[XYZ]) -> Tensor[YZ]:
    """Project XYZ tensor to YZ by reducing X dimension (axis 0)."""
    return Tensor[YZ](graph.reduce_mean(t.t, axis=0))
