"""
Common Tensor Spaces
====================

Pre-defined tensor spaces for common use cases. This module provides
ready-to-use tensor spaces to avoid repeated instantiation.

Note: This module uses mathematical terminology (bases, dimensions,
projections) to provide intuitive abstractions for working with
multidimensional arrays. While inspired by vector space concepts,
these are engineering approximations rather than rigorous
mathematical constructs.

Examples:
--------
    >>> from yakof.frontend import spaces
    >>> x = spaces.x.placeholder("x")  # tensor in R¹(X)
    >>> xy = spaces.xy.placeholder("xy")  # tensor in R²(X,Y)
    >>> xyz = spaces.xyz.placeholder("xyz")  # tensor in R³(X,Y,Z)
"""

from yakof.frontend import abstract, bases

# R¹ spaces
x = abstract.TensorSpace[bases.X]()
y = abstract.TensorSpace[bases.Y]()
z = abstract.TensorSpace[bases.Z]()

# R² spaces
xy = abstract.TensorSpace[bases.XY]()
xz = abstract.TensorSpace[bases.XZ]()
yz = abstract.TensorSpace[bases.YZ]()

# R³ space
xyz = abstract.TensorSpace[bases.XYZ]()
