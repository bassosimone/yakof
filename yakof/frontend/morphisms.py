"""
Morphisms Between Tensor Spaces.
===============================

This module implements morphisms between tensor spaces, allowing tensors to be
transformed between spaces of different dimensions.

Classes
-------

ExpandDims
    The primary class for expanding tensors to higher dimensional spaces by
    inserting new dimensions in the correct positions.

ProjectUsingSum
    The primary class for projecting tensors to lower dimensional spaces by
    summing over removed dimensions.

Low-Level Functions
-------------------

axes_expansion
    Internal function for calculating positions where new dimensions should
    be inserted. Most users should use ExpandDims instead.

axes_projection
    Internal function for calculating which dimensions to sum over. Most users
    should use ProjectUsingSum instead.

generate_canonical_axes
    Generates canonical axes IDs for a 1-dimensional tensor space of given size.

Implementation Details
----------------------

The implementation relies on three key properties:

1. Canonical axes: Each dimension has a unique integer ID (e.g., X=1000,
Y=1001, Z=1002). This establishes a consistent ordering and makes
debugging easier by clearly distinguishing axes from array indices.

2. Monotonicity: Dimensions are always used in ascending order of their
IDs. Both source and destination spaces must maintain this ordering.

3. Subset relationships: For expansion, source axes must be a subset of
destination axes. For projection, destination axes must be a subset of
source axes.

Examples
--------
Expand a 1D tensor in Z space to a 2D tensor in YZ space:

    >>> expand = ExpandDims(Z, YZ)  # Z ⊆ YZ
    >>> yz_tensor = expand(z_tensor)  # Adds Y dimension at index 0

Project a 3D tensor in XYZ space to a 2D tensor in XZ space:

    >>> project = ProjectUsingSum(XYZ, XZ)  # XZ ⊆ XYZ
    >>> xz_tensor = project(xyz_tensor)  # Sums over Y dimension
"""

from typing import Generic, TypeVar

from . import abstract, graph


def generate_canonical_axes(size: int) -> tuple[int, ...]:
    """Generates axes IDs for the 1-dimensional tensors defining a R^size space.

    This provides a convenient way to get consistent dimension IDs for
    a space of given size. The IDs will be equally spaced monotonically
    increasing integers and there is no guarantee that they will be
    starting from zero. (Actually, using numbers around zero makes it
    very hard when debugging to set apart indexes and axes IDs.)

    Examples
    --------
        >>> generate_canonical_axes(3)
        (1000, 1001, 1002)
        >>> generate_canonical_axes(2)
        (1000, 1001)
    """
    return tuple(x + 1000 for x in range(size))


def axes_expansion(source: graph.Axis, dest: graph.Axis) -> graph.Axis:
    """Calculate axes for expanding from source to destination space.

    .. warning::
        This is a low-level function. Most users should use ExpandDims instead.

    Determines which axes to insert when expanding a tensor from the source
    space to the destination space. Handles single axis and multiple axes.

    Args:
        source: Dimension(s) in source space
        dest: Dimension(s) in destination space

    Returns
    -------
        Single axis position if only one dimension needs to be inserted,
        otherwise tuple of positions for inserting new dimensions

    Raises
    ------
        ValueError: if source is not a subset of dest.
        ValueError: if source values are not monotonically increasing.
        ValueError: if dest values are not monotonically increasing.

    Examples
    --------
        >>> x, y, z = generate_canonical_axes(3)  # Get distinct axis IDs
        >>> axes_expansion((x,), (x, y))
        1  # Position to insert Y
        >>> axes_expansion((x,), (x, y, z))
        (1, 2)  # Positions to insert Y and Z

    See Also
    --------
        ExpandDims: The high-level API for dimension expansion
    """
    # Handle single axis case
    source = source if isinstance(source, tuple) else (source,)
    dest = dest if isinstance(dest, tuple) else (dest,)

    # Ensure that the destination is a subset of the source
    if not set(source).issubset(set(dest)):
        raise ValueError("source must be a subset of destination")

    # Ensure that source has monotonic values
    if source != tuple(sorted(source)):
        raise ValueError("source must have monotonic values")

    # Ensure that dest has monotonic values
    if dest != tuple(sorted(dest)):
        raise ValueError("dest must have monotonic values")

    # Obtain the indexes of the dest values not belonging to source
    rv = [idx for idx, value in enumerate(dest) if value not in set(source)]

    # Return single axis or tuple based on result size
    return rv[0] if len(rv) == 1 else tuple(rv)


def axes_projection(source: graph.Axis, dest: graph.Axis) -> graph.Axis:
    """Calculate axes for projecting from source to destination space.

    .. warning::
        This is a low-level function. Most users should use ProjectUsingSum instead.

    Determines which axes to sum over when projecting a tensor from
    the source space to the destination space. Handles both single
    axis and multiple axes cases.

    Args:
        source: Dimension(s) in source space
        dest: Dimension(s) in destination space

    Returns
    -------
        Single axis to sum over if only one dimension needs to be removed,
        otherwise tuple of axes to sum over

    Raises
    ------
        ValueError: if dest is not a subset of source.
        ValueError: if source values are not monotonically increasing.
        ValueError: if dest values are not monotonically increasing.

    Examples
    --------
        >>> x, y, z = generate_canonical_axes(3)  # Get distinct axis IDs
        >>> axes_projection((x, y, z), (x, z))
        1  # Position of Y to sum over
        >>> axes_projection((x, y, z), x)
        (1, 2)  # Positions of Y and Z to sum over
        >>> axes_projection((x, y, z), (y, z))
        0  # Position of X to sum over

    See Also
    --------
        ProjectUsingSum: The high-level API for dimension reduction
    """
    # Handle single axis case
    source = source if isinstance(source, tuple) else (source,)
    dest = dest if isinstance(dest, tuple) else (dest,)

    # Ensure that the source is a subset of the destination
    if not set(dest).issubset(set(source)):
        raise ValueError("destination must be a subset of source")

    # Ensure that source has monotonic values
    if source != tuple(sorted(source)):
        raise ValueError("source must have monotonic values")

    # Ensure that dest has monotonic values
    if dest != tuple(sorted(dest)):
        raise ValueError("dest must have monotonic values")

    # Obtain the indexes of the source values not belonging to dest
    rv = [idx for idx, value in enumerate(source) if value not in set(dest)]

    # Return single axis or tuple based on result size
    return rv[0] if len(rv) == 1 else tuple(rv)


A = TypeVar("A")
"""Type variable for source tensor space."""

B = TypeVar("B")
"""Type variable for destination tensor space."""


class ExpandDims(Generic[A, B]):
    """Morphism that expands tensors to higher dimensional spaces.

    Type Parameters:
        A: Source basis type
        B: Destination basis type

    Args:
        source: source tensor space.
        dest: destination tensor space.

    Attributes
    ----------
        source: source tensor space.
        dest: destination tensor space.

    Example:
        >>> expand = ExpandDims(space_z, space_yz)
        >>> yz_tensor = expand(z_tensor)  # Expands 1D to 2D
    """

    def __init__(self, source: abstract.TensorSpace[A], dest: abstract.TensorSpace[B]):
        self.source = source
        self.dest = dest

    def __call__(self, t: abstract.Tensor[A]) -> abstract.Tensor[B]:
        """Apply the expansion morphism to a tensor.

        Calculates required axes and uses numpy's expand_dims to insert
        new dimensions in the correct positions.
        """
        axes = axes_expansion(self.source.axes(), self.dest.axes())
        return self.dest.new_tensor(graph.expand_dims(t.node, axis=axes))


class ProjectUsingSum(Generic[A, B]):
    """Morphism that projects tensors to lower dimensional spaces using summation.

    Type Parameters:
        A: Source basis type
        B: Destination basis type

    Args:
        source: source tensor space.
        dest: destination tensor space.

    Attributes
    ----------
        source: source tensor space.
        dest: destination tensor space.

    Example:
        >>> project = ProjectUsingSum(space_xyz, space_xz)
        >>> xz_tensor = project(xyz_tensor)  # Projects 3D to 2D
    """

    def __init__(self, source: abstract.TensorSpace[A], dest: abstract.TensorSpace[B]):
        self.source = source
        self.dest = dest

    def __call__(self, t: abstract.Tensor[A]) -> abstract.Tensor[B]:
        """Apply the projection morphism to a tensor.

        Calculates required axes and uses numpy's reduce_sum to eliminate
        dimensions by summing over them.
        """
        axes = axes_projection(self.source.axes(), self.dest.axes())
        return self.dest.new_tensor(graph.reduce_sum(t.node, axis=axes))
