"""
Common Tensor Bases
===================

This module provides pre-defined bases for common tensor spaces
up to R^6. Each class in this module implements the Basis protocol
defined by the abstract module of this package.

Single axes are named using single letters (X, Y, Z, U, V, W).
Compound bases use concatenated letters (XY, XYZ, etc.).

Example
-------

    >>> from yakof.frontend import abstract, bases
    >>>
    >>> space = abstract.TensorSpace(bases.X)
    >>> a = space.placeholder("a")  # tensor in R^1


Axis Numbering Convention
-------------------------

- Axes 0-2: Physical space (X, Y, Z)
- Axes 3-5: Additional dimensions (U, V, W)
- Compound bases preserve ordering
- No gaps allowed in axis sequences

This scheme allows:
- Natural mapping to physical quantities
- Consistent handling of projections
- Clear dimensional relationships
"""

from yakof.frontend import graph


# Single axes (R^1)
class X:
    @staticmethod
    def axis() -> graph.Axis:
        return 0


class Y:
    @staticmethod
    def axis() -> graph.Axis:
        return 1


class Z:
    @staticmethod
    def axis() -> graph.Axis:
        return 2


class U:
    @staticmethod
    def axis() -> graph.Axis:
        return 3


class V:
    @staticmethod
    def axis() -> graph.Axis:
        return 4


class W:
    @staticmethod
    def axis() -> graph.Axis:
        return 5


# R^2 combinations
class XY:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1)


class XZ:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 2)


class YZ:
    @staticmethod
    def axis() -> graph.Axis:
        return (1, 2)


class UV:
    @staticmethod
    def axis() -> graph.Axis:
        return (3, 4)


class UW:
    @staticmethod
    def axis() -> graph.Axis:
        return (3, 5)


class VW:
    @staticmethod
    def axis() -> graph.Axis:
        return (4, 5)


# R^3 combinations
class XYZ:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2)


class UVW:
    @staticmethod
    def axis() -> graph.Axis:
        return (3, 4, 5)


# R^4 combinations
class XYZU:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2, 3)


class XYZV:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2, 4)


class XYZW:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2, 5)


# R^5 combinations
class XYZUV:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2, 3, 4)


class XYZUW:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2, 3, 5)


class XYZVW:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2, 4, 5)


# R^6
class XYZUVW:
    @staticmethod
    def axis() -> graph.Axis:
        return (0, 1, 2, 3, 4, 5)
