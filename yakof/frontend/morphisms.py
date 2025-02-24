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
    a space of given size. The IDs will be 0,1,2,...,size-1.
    """
    return tuple(range(size))


sourceflag, destflag = 1 << 0, 1 << 1


def _morph_with_predicate(
    source: graph.Axis,
    dest: graph.Axis,
    predicate: Callable[[int], bool],
) -> graph.Axis:
    # Handle single axis case
    source = source if isinstance(source, tuple) else (source,)
    dest = dest if isinstance(dest, tuple) else (dest,)

    # Build map of dimension -> flags
    merged = {}
    for axis in source:
        merged[axis] = merged.get(axis, 0) | sourceflag
    for axis in dest:
        merged[axis] = merged.get(axis, 0) | destflag

    # Filter the positions we should return
    rv = []
    for idx, axis in enumerate(sorted(merged.keys())):
        if predicate(merged[axis]):
            rv.append(idx)

    # Return single axis or tuple based on result size
    res = tuple(rv)
    return res[0] if len(res) == 1 else res


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

    See Also:
        ExpandDims: The high-level API for dimension expansion
    """
    return _morph_with_predicate(source, dest, lambda x: x & sourceflag == 0)


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

    See Also:
        ProjectUsingSum: The high-level API for dimension reduction
    """
    return _morph_with_predicate(source, dest, lambda x: x & destflag == 0)


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
