"""
Morphisms Between Tensor Spaces
===============================

This module implements morphisms between tensor spaces, allowing tensors to be
transformed between spaces of different dimensions.

Main Classes
------------

ExpandDims
    The primary class for expanding tensors to higher dimensional spaces.
    This should be your default choice for dimension expansion.

ProjectUsingSum
    The primary class for projecting tensors to lower dimensional spaces.
    This should be your default choice for dimension reduction.

Low-Level Functions
-------------------

axes_expansion
    Internal function for calculating expansion axes. Most users should use
    ExpandDims instead.

axes_projection
    Internal function for calculating projection axes. Most users should use
    ProjectUsingSum instead.

Implementation Details
----------------------

The implementation relies on three key properties:

1. Consistent dimension numbering: Each dimension has a unique small integer ID
   (e.g., X=0, Y=1, Z=2). This establishes a canonical ordering.

2. Canonical ordering: Dimensions are always used in ascending order of their IDs.
   This matches both numpy's dimension ordering and our semantic needs.

3. numpy's dimension operations: The implementation matches numpy's behavior for
   inserting and reducing dimensions, particularly:
   - expand_dims: Inserts new axes at specified positions
   - reduce_sum: Reduces (sums) along specified axes

Examples

--------
To expand a tensor from space Z to space YZ:

    >>> expand = ExpandDims(Z, YZ)
    >>> yz_tensor = expand(z_tensor)

To project a tensor from space XYZ to space XZ:

    >>> project = ProjectUsingSum(XYZ, XZ)
    >>> xz_tensor = project(xyz_tensor)
"""

from typing import Callable, Protocol, runtime_checkable

from . import abstract, graph


def generate_canonical_axes(size: int) -> tuple[int, ...]:
    """Generates axes IDs for the 1-dimensional tensors defining a R^size space.

    This provides a convenient way to get consistent dimension IDs for
    a space of given size. The IDs will be equally spaced monotonically
    increasing integers and there is no guarantee that they will be
    starting from zero. (Actually, using numbers around zero makes it
    very hard when debugging to set apart indexes and axes IDs.)
    """
    return tuple(x + 1000 for x in range(size))


def axes_expansion(source: graph.Axis, dest: graph.Axis) -> graph.Axis:
    """Calculate axes for expanding from source to destination space.

    .. warning::
        This is a low-level function. Most users should use ExpandDims instead.

    Determines positions where new dimensions should be inserted when expanding
    a tensor from the source space to the destination space. Handles both single
    axis and multiple axes cases.

    Args:
        source: Dimension(s) in source space
        dest: Dimension(s) in destination space

    Returns:
        Single axis position if only one dimension needs to be inserted,
        otherwise tuple of positions for inserting new dimensions

    Raises:
        ValueError: if dest is smaller than source.

    See Also:
        ExpandDims: The high-level API for dimension expansion
    """
    # Handle single axis case
    source = source if isinstance(source, tuple) else (source,)
    dest = dest if isinstance(dest, tuple) else (dest,)

    # TODO: raise if dest is smaller than source!!!

    # TODO: raise if tuples do not monotonically increase!!!

    # TODO: raise if they are disjoint, that is the second set MUST
    # be a superset of the original set!!!

    # To understand the algorithm, consider this example. We want to
    # expand (X, Z, V) to (X, Y, Z, U, V). We also assume that the axes
    # are numbered from 10 to 14. The source axes are (10, 12, 14) and
    # the destination axes are (10, 11, 12, 13, 14). We're using numbers
    # starting from 10 to illustrate a more general case compared to
    # the one where we're starting to number from zero.
    #
    # TODO: verify if we have more restrictions here! Ideally we would
    # like that this mehcanism works with as little restrictions as possible.
    #
    # We build a map that tracks which axes appear in input and/or
    # output, by using flags to indicate the presence of an axis. In
    # the above example, we end up with the following map:
    #
    #   {
    #       10: sourceflag | destflag,
    #       11: destflag,
    #       12: sourceflag | destflag,
    #       13: destflag,
    #       14: sourceflag | destflag,
    #   }
    #
    # Note that we are using the axes numbers as the map keys.
    sourceflag, destflag = 1 << 0, 1 << 1
    merged = {}
    for axis in source:
        merged[axis] = merged.get(axis, 0) | sourceflag
    for axis in dest:
        merged[axis] = merged.get(axis, 0) | destflag

    # Maps keys are ordered with random order but we can sort them
    # and follow canonical ordering to determine the positions where
    # new dimensions should be inserted. In other words, first we
    # build this data structure:
    #
    #   (
    #       (0, 10),  # sourceflag | destflag
    #       (1, 11),  # destflag
    #       (2, 12),  # sourceflag | destflag
    #       (3, 13),  # destflag
    #       (4, 14),  # sourceflag | destflag
    #   )
    #
    # Comments on the side indicate the corresponding map values.
    #
    # Then, we filter to only include the indexes of the axes that do
    # not appear in the source. In this example, we get:
    #
    #   (1, 3)
    #
    # This means we will invoke np.expand_dims as follows:
    #
    #   np.expand_dims(..., axis=(1, 3))
    #
    # That is, we add axes in positions 1 and 3.
    rv = []
    for idx, axis in enumerate(sorted(merged.keys())):
        if merged[axis] == destflag:
            rv.append(idx)

    # Return single axis or tuple based on result size
    res = tuple(rv)
    return res[0] if len(res) == 1 else res


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

    Returns:
        Single axis to sum over if only one dimension needs to be removed,
        otherwise tuple of axes to sum over

    Raises:
        ValueError: if dest is not a subset of source.
        ValueError: if source values are not monotonically increasing.
        ValueError: if dest values are not monotonically increasing.

    Examples:
        >>> x, y, z = generate_canonical_axes(3)  # Get distinct axis IDs
        >>> axes_projection((x, y, z), (x, z))
        1  # Position of Y to sum over
        >>> axes_projection((x, y, z), x)
        (1, 2)  # Positions of Y and Z to sum over
        >>> axes_projection((x, y, z), (y, z))
        0  # Position of X to sum over

    See Also:
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


@runtime_checkable
class Basis(Protocol):
    """Protocol defining the interface for tensor space bases.

    All bases must provide their axes as a set of integers, establishing
    their position in the canonical ordering. To generate the canonical
    ordering, use the generate_canonical_axes function.
    """

    axes: set[int]


class ExpandDims[A: Basis, B: Basis]:
    """Morphism that expands tensors to higher dimensional spaces.

    Type Parameters:
        A: Source basis type
        B: Destination basis type

    Args:
        source: Instance of A
        dest: Instance of B

    Example:
        >>> expand = ExpandDims(Z, YZ)
        >>> yz_tensor = expand(z_tensor)
    """

    def __init__(self, source: type[A], dest: type[B]):
        self.source = source
        self.dest = dest

    def __call__(self, t: abstract.Tensor[A]) -> abstract.Tensor[B]:
        """Apply the expansion morphism to a tensor.

        Calculates required axes and uses numpy's expand_dims to insert
        new dimensions in the correct positions.
        """
        axes = axes_expansion(tuple(self.source.axes), tuple(self.dest.axes))
        return abstract.Tensor[B](graph.expand_dims(t.node, axis=axes))


class ProjectUsingSum[A: Basis, B: Basis]:
    """Morphism that projects tensors to lower dimensional spaces using summation.

    Type Parameters:
        A: Source basis type
        B: Destination basis type

    Args:
        source: Instance of A
        dest: Instance of B

    Example:
        >>> project = ProjectUsingSum(XYZ, XZ)
        >>> xz_tensor = project(xyz_tensor)
    """

    def __init__(self, source: type[A], dest: type[B]):
        self.source = source
        self.dest = dest

    def __call__(self, t: abstract.Tensor[A]) -> abstract.Tensor[B]:
        """Apply the projection morphism to a tensor.

        Calculates required axes and uses numpy's reduce_sum to eliminate
        dimensions by summing over them.
        """
        axes = axes_projection(tuple(self.source.axes), tuple(self.dest.axes))
        return abstract.Tensor[B](graph.reduce_sum(t.node, axis=axes))
