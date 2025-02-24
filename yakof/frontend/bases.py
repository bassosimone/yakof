"""
Tensor Bases and Transformations
================================

This module provides type definitions and transformations for tensor spaces
up to R⁶, using the canonical dimensions X, Y, Z, U, V, and W.

Each basis defines its axes as a set of integers following the canonical ordering:
X=0, Y=1, Z=2, U=3, V=4, W=5. These are used by the morphism classes to
automatically compute the correct axes for expansions and projections.

Type Definitions:
---------------
- R¹: X, Y, Z, U, V, W (single axes)
- R²: XY, XZ, YZ, ... (15 pairs of axes)
- R³: XYZ, XYU, ... (20 combinations)
- R⁴: XYZU, ... (15 combinations)
- R⁵: XYZUV, ... (6 combinations)
- R⁶: XYZUVW (1 combination)

Expansions:
----------
- R¹ -> R²: 30 morphisms (each single axis to its valid 2D combinations)
- R² -> R³: 60 morphisms (each 2D space to its valid 3D combinations)
- R³ -> R⁴: 60 morphisms (each 3D space to its valid 4D combinations)
- R⁴ -> R⁵: 30 morphisms (each 4D space to its valid 5D combinations)
- R⁵ -> R⁶: 6 morphisms (each 5D space to the full 6D space)

Projections:
-----------
- R² -> R¹: 30 morphisms (each 2D space to its constituent axes)
- R³ -> R²: 60 morphisms (each 3D space to its valid 2D combinations)
- R⁴ -> R³: 60 morphisms (each 4D space to its valid 3D combinations)
- R⁵ -> R⁴: 30 morphisms (each 5D space to its valid 4D combinations)
- R⁶ -> R⁵: 6 morphisms (full 6D space to each 5D combination)

Example:
-------
    >>> from yakof.frontend import abstract, bases, morphisms
    >>> space_x = abstract.TensorSpace[bases.X]()
    >>> x = space_x.placeholder("x")
    >>> expand = morphisms.ExpandDims(bases.X, bases.XY)
    >>> xy = expand(x)
"""

from yakof.frontend import morphisms


# R¹ bases
class X:
    """Tensor basis along the X axis."""

    axes = {0}


class Y:
    """Tensor basis along the Y axis."""

    axes = {1}


class Z:
    """Tensor basis along the Z axis."""

    axes = {2}


class U:
    """Tensor basis along the U axis."""

    axes = {3}


class V:
    """Tensor basis along the V axis."""

    axes = {4}


class W:
    """Tensor basis along the W axis."""

    axes = {5}


# R² bases
class XY:
    """Tensor basis along X and Y axes."""

    axes = {0, 1}


class XZ:
    """Tensor basis along X and Z axes."""

    axes = {0, 2}


class XU:
    """Tensor basis along X and U axes."""

    axes = {0, 3}


class XV:
    """Tensor basis along X and V axes."""

    axes = {0, 4}


class XW:
    """Tensor basis along X and W axes."""

    axes = {0, 5}


class YZ:
    """Tensor basis along Y and Z axes."""

    axes = {1, 2}


class YU:
    """Tensor basis along Y and U axes."""

    axes = {1, 3}


class YV:
    """Tensor basis along Y and V axes."""

    axes = {1, 4}


class YW:
    """Tensor basis along Y and W axes."""

    axes = {1, 5}


class ZU:
    """Tensor basis along Z and U axes."""

    axes = {2, 3}


class ZV:
    """Tensor basis along Z and V axes."""

    axes = {2, 4}


class ZW:
    """Tensor basis along Z and W axes."""

    axes = {2, 5}


class UV:
    """Tensor basis along U and V axes."""

    axes = {3, 4}


class UW:
    """Tensor basis along U and W axes."""

    axes = {3, 5}


class VW:
    """Tensor basis along V and W axes."""

    axes = {4, 5}


# R³ bases
class XYZ:
    """Tensor basis along X, Y and Z axes."""

    axes = {0, 1, 2}


class XYU:
    """Tensor basis along X, Y and U axes."""

    axes = {0, 1, 3}


class XYV:
    """Tensor basis along X, Y and V axes."""

    axes = {0, 1, 4}


class XYW:
    """Tensor basis along X, Y and W axes."""

    axes = {0, 1, 5}


class XZU:
    """Tensor basis along X, Z and U axes."""

    axes = {0, 2, 3}


class XZV:
    """Tensor basis along X, Z and V axes."""

    axes = {0, 2, 4}


class XZW:
    """Tensor basis along X, Z and W axes."""

    axes = {0, 2, 5}


class XUV:
    """Tensor basis along X, U and V axes."""

    axes = {0, 3, 4}


class XUW:
    """Tensor basis along X, U and W axes."""

    axes = {0, 3, 5}


class XVW:
    """Tensor basis along X, V and W axes."""

    axes = {0, 4, 5}


class YZU:
    """Tensor basis along Y, Z and U axes."""

    axes = {1, 2, 3}


class YZV:
    """Tensor basis along Y, Z and V axes."""

    axes = {1, 2, 4}


class YZW:
    """Tensor basis along Y, Z and W axes."""

    axes = {1, 2, 5}


class YUV:
    """Tensor basis along Y, U and V axes."""

    axes = {1, 3, 4}


class YUW:
    """Tensor basis along Y, U and W axes."""

    axes = {1, 3, 5}


class YVW:
    """Tensor basis along Y, V and W axes."""

    axes = {1, 4, 5}


class ZUV:
    """Tensor basis along Z, U and V axes."""

    axes = {2, 3, 4}


class ZUW:
    """Tensor basis along Z, U and W axes."""

    axes = {2, 3, 5}


class ZVW:
    """Tensor basis along Z, V and W axes."""

    axes = {2, 4, 5}


class UVW:
    """Tensor basis along U, V and W axes."""

    axes = {3, 4, 5}


# R⁴ bases
class XYZU:
    """Tensor basis along X, Y, Z and U axes."""

    axes = {0, 1, 2, 3}


class XYZV:
    """Tensor basis along X, Y, Z and V axes."""

    axes = {0, 1, 2, 4}


class XYZW:
    """Tensor basis along X, Y, Z and W axes."""

    axes = {0, 1, 2, 5}


class XYUV:
    """Tensor basis along X, Y, U and V axes."""

    axes = {0, 1, 3, 4}


class XYUW:
    """Tensor basis along X, Y, U and W axes."""

    axes = {0, 1, 3, 5}


class XYVW:
    """Tensor basis along X, Y, V and W axes."""

    axes = {0, 1, 4, 5}


class XZUV:
    """Tensor basis along X, Z, U and V axes."""

    axes = {0, 2, 3, 4}


class XZUW:
    """Tensor basis along X, Z, U and W axes."""

    axes = {0, 2, 3, 5}


class XZVW:
    """Tensor basis along X, Z, V and W axes."""

    axes = {0, 2, 4, 5}


class XUVW:
    """Tensor basis along X, U, V and W axes."""

    axes = {0, 3, 4, 5}


class YZUV:
    """Tensor basis along Y, Z, U and V axes."""

    axes = {1, 2, 3, 4}


class YZUW:
    """Tensor basis along Y, Z, U and W axes."""

    axes = {1, 2, 3, 5}


class YZVW:
    """Tensor basis along Y, Z, V and W axes."""

    axes = {1, 2, 4, 5}


class YUVW:
    """Tensor basis along Y, U, V and W axes."""

    axes = {1, 3, 4, 5}


class ZUVW:
    """Tensor basis along Z, U, V and W axes."""

    axes = {2, 3, 4, 5}


# R⁵ bases
class XYZUV:
    """Tensor basis along X, Y, Z, U and V axes."""

    axes = {0, 1, 2, 3, 4}


class XYZUW:
    """Tensor basis along X, Y, Z, U and W axes."""

    axes = {0, 1, 2, 3, 5}


class XYZVW:
    """Tensor basis along X, Y, Z, V and W axes."""

    axes = {0, 1, 2, 4, 5}


class XYUVW:
    """Tensor basis along X, Y, U, V and W axes."""

    axes = {0, 1, 3, 4, 5}


class XZUVW:
    """Tensor basis along X, Z, U, V and W axes."""

    axes = {0, 2, 3, 4, 5}


class YZUVW:
    """Tensor basis along Y, Z, U, V and W axes."""

    axes = {1, 2, 3, 4, 5}


# R⁶ basis
class XYZUVW:
    """Tensor basis along X, Y, Z, U, V and W axes."""

    axes = {0, 1, 2, 3, 4, 5}


# R¹ -> R² expansions
expand_x_to_xy = morphisms.ExpandDims(X, XY)
expand_x_to_xz = morphisms.ExpandDims(X, XZ)
expand_x_to_xu = morphisms.ExpandDims(X, XU)
expand_x_to_xv = morphisms.ExpandDims(X, XV)
expand_x_to_xw = morphisms.ExpandDims(X, XW)
expand_y_to_xy = morphisms.ExpandDims(Y, XY)
expand_y_to_yz = morphisms.ExpandDims(Y, YZ)
expand_y_to_yu = morphisms.ExpandDims(Y, YU)
expand_y_to_yv = morphisms.ExpandDims(Y, YV)
expand_y_to_yw = morphisms.ExpandDims(Y, YW)
expand_z_to_xz = morphisms.ExpandDims(Z, XZ)
expand_z_to_yz = morphisms.ExpandDims(Z, YZ)
expand_z_to_zu = morphisms.ExpandDims(Z, ZU)
expand_z_to_zv = morphisms.ExpandDims(Z, ZV)
expand_z_to_zw = morphisms.ExpandDims(Z, ZW)
expand_u_to_xu = morphisms.ExpandDims(U, XU)
expand_u_to_yu = morphisms.ExpandDims(U, YU)
expand_u_to_zu = morphisms.ExpandDims(U, ZU)
expand_u_to_uv = morphisms.ExpandDims(U, UV)
expand_u_to_uw = morphisms.ExpandDims(U, UW)
expand_v_to_xv = morphisms.ExpandDims(V, XV)
expand_v_to_yv = morphisms.ExpandDims(V, YV)
expand_v_to_zv = morphisms.ExpandDims(V, ZV)
expand_v_to_uv = morphisms.ExpandDims(V, UV)
expand_v_to_vw = morphisms.ExpandDims(V, VW)
expand_w_to_xw = morphisms.ExpandDims(W, XW)
expand_w_to_yw = morphisms.ExpandDims(W, YW)
expand_w_to_zw = morphisms.ExpandDims(W, ZW)
expand_w_to_uw = morphisms.ExpandDims(W, UW)
expand_w_to_vw = morphisms.ExpandDims(W, VW)

# R² -> R³ expansions
expand_xy_to_xyz = morphisms.ExpandDims(XY, XYZ)
expand_xy_to_xyu = morphisms.ExpandDims(XY, XYU)
expand_xy_to_xyv = morphisms.ExpandDims(XY, XYV)
expand_xy_to_xyw = morphisms.ExpandDims(XY, XYW)
expand_xz_to_xyz = morphisms.ExpandDims(XZ, XYZ)
expand_xz_to_xzu = morphisms.ExpandDims(XZ, XZU)
expand_xz_to_xzv = morphisms.ExpandDims(XZ, XZV)
expand_xz_to_xzw = morphisms.ExpandDims(XZ, XZW)
expand_xu_to_xyu = morphisms.ExpandDims(XU, XYU)
expand_xu_to_xzu = morphisms.ExpandDims(XU, XZU)
expand_xu_to_xuv = morphisms.ExpandDims(XU, XUV)
expand_xu_to_xuw = morphisms.ExpandDims(XU, XUW)
expand_xv_to_xyv = morphisms.ExpandDims(XV, XYV)
expand_xv_to_xzv = morphisms.ExpandDims(XV, XZV)
expand_xv_to_xuv = morphisms.ExpandDims(XV, XUV)
expand_xv_to_xvw = morphisms.ExpandDims(XV, XVW)
expand_xw_to_xyw = morphisms.ExpandDims(XW, XYW)
expand_xw_to_xzw = morphisms.ExpandDims(XW, XZW)
expand_xw_to_xuw = morphisms.ExpandDims(XW, XUW)
expand_xw_to_xvw = morphisms.ExpandDims(XW, XVW)
expand_yz_to_xyz = morphisms.ExpandDims(YZ, XYZ)
expand_yz_to_yzu = morphisms.ExpandDims(YZ, YZU)
expand_yz_to_yzv = morphisms.ExpandDims(YZ, YZV)
expand_yz_to_yzw = morphisms.ExpandDims(YZ, YZW)
expand_yu_to_xyu = morphisms.ExpandDims(YU, XYU)
expand_yu_to_yzu = morphisms.ExpandDims(YU, YZU)
expand_yu_to_yuv = morphisms.ExpandDims(YU, YUV)
expand_yu_to_yuw = morphisms.ExpandDims(YU, YUW)
expand_yv_to_xyv = morphisms.ExpandDims(YV, XYV)
expand_yv_to_yzv = morphisms.ExpandDims(YV, YZV)
expand_yv_to_yuv = morphisms.ExpandDims(YV, YUV)
expand_yv_to_yvw = morphisms.ExpandDims(YV, YVW)
expand_yw_to_xyw = morphisms.ExpandDims(YW, XYW)
expand_yw_to_yzw = morphisms.ExpandDims(YW, YZW)
expand_yw_to_yuw = morphisms.ExpandDims(YW, YUW)
expand_yw_to_yvw = morphisms.ExpandDims(YW, YVW)
expand_zu_to_xzu = morphisms.ExpandDims(ZU, XZU)
expand_zu_to_yzu = morphisms.ExpandDims(ZU, YZU)
expand_zu_to_zuv = morphisms.ExpandDims(ZU, ZUV)
expand_zu_to_zuw = morphisms.ExpandDims(ZU, ZUW)
expand_zv_to_xzv = morphisms.ExpandDims(ZV, XZV)
expand_zv_to_yzv = morphisms.ExpandDims(ZV, YZV)
expand_zv_to_zuv = morphisms.ExpandDims(ZV, ZUV)
expand_zv_to_zvw = morphisms.ExpandDims(ZV, ZVW)
expand_zw_to_xzw = morphisms.ExpandDims(ZW, XZW)
expand_zw_to_yzw = morphisms.ExpandDims(ZW, YZW)
expand_zw_to_zuw = morphisms.ExpandDims(ZW, ZUW)
expand_zw_to_zvw = morphisms.ExpandDims(ZW, ZVW)
expand_uv_to_xuv = morphisms.ExpandDims(UV, XUV)
expand_uv_to_yuv = morphisms.ExpandDims(UV, YUV)
expand_uv_to_zuv = morphisms.ExpandDims(UV, ZUV)
expand_uv_to_uvw = morphisms.ExpandDims(UV, UVW)
expand_uw_to_xuw = morphisms.ExpandDims(UW, XUW)
expand_uw_to_yuw = morphisms.ExpandDims(UW, YUW)
expand_uw_to_zuw = morphisms.ExpandDims(UW, ZUW)
expand_uw_to_uvw = morphisms.ExpandDims(UW, UVW)
expand_vw_to_xvw = morphisms.ExpandDims(VW, XVW)
expand_vw_to_yvw = morphisms.ExpandDims(VW, YVW)
expand_vw_to_zvw = morphisms.ExpandDims(VW, ZVW)
expand_vw_to_uvw = morphisms.ExpandDims(VW, UVW)

# R³ -> R⁴ expansions
expand_xyz_to_xyzu = morphisms.ExpandDims(XYZ, XYZU)
expand_xyz_to_xyzv = morphisms.ExpandDims(XYZ, XYZV)
expand_xyz_to_xyzw = morphisms.ExpandDims(XYZ, XYZW)
expand_xyu_to_xyzu = morphisms.ExpandDims(XYU, XYZU)
expand_xyu_to_xyuv = morphisms.ExpandDims(XYU, XYUV)
expand_xyu_to_xyuw = morphisms.ExpandDims(XYU, XYUW)
expand_xyv_to_xyzv = morphisms.ExpandDims(XYV, XYZV)
expand_xyv_to_xyuv = morphisms.ExpandDims(XYV, XYUV)
expand_xyv_to_xyvw = morphisms.ExpandDims(XYV, XYVW)
expand_xyw_to_xyzw = morphisms.ExpandDims(XYW, XYZW)
expand_xyw_to_xyuw = morphisms.ExpandDims(XYW, XYUW)
expand_xyw_to_xyvw = morphisms.ExpandDims(XYW, XYVW)
expand_xzu_to_xyzu = morphisms.ExpandDims(XZU, XYZU)
expand_xzu_to_xzuv = morphisms.ExpandDims(XZU, XZUV)
expand_xzu_to_xzuw = morphisms.ExpandDims(XZU, XZUW)
expand_xzv_to_xyzv = morphisms.ExpandDims(XZV, XYZV)
expand_xzv_to_xzuv = morphisms.ExpandDims(XZV, XZUV)
expand_xzv_to_xzvw = morphisms.ExpandDims(XZV, XZVW)
expand_xzw_to_xyzw = morphisms.ExpandDims(XZW, XYZW)
expand_xzw_to_xzuw = morphisms.ExpandDims(XZW, XZUW)
expand_xzw_to_xzvw = morphisms.ExpandDims(XZW, XZVW)
expand_xuv_to_xyuv = morphisms.ExpandDims(XUV, XYUV)
expand_xuv_to_xzuv = morphisms.ExpandDims(XUV, XZUV)
expand_xuv_to_xuvw = morphisms.ExpandDims(XUV, XUVW)
expand_xuw_to_xyuw = morphisms.ExpandDims(XUW, XYUW)
expand_xuw_to_xzuw = morphisms.ExpandDims(XUW, XZUW)
expand_xuw_to_xuvw = morphisms.ExpandDims(XUW, XUVW)
expand_xvw_to_xyvw = morphisms.ExpandDims(XVW, XYVW)
expand_xvw_to_xzvw = morphisms.ExpandDims(XVW, XZVW)
expand_xvw_to_xuvw = morphisms.ExpandDims(XVW, XUVW)
expand_yzu_to_xyzu = morphisms.ExpandDims(YZU, XYZU)
expand_yzu_to_yzuv = morphisms.ExpandDims(YZU, YZUV)
expand_yzu_to_yzuw = morphisms.ExpandDims(YZU, YZUW)
expand_yzv_to_xyzv = morphisms.ExpandDims(YZV, XYZV)
expand_yzv_to_yzuv = morphisms.ExpandDims(YZV, YZUV)
expand_yzv_to_yzvw = morphisms.ExpandDims(YZV, YZVW)
expand_yzw_to_xyzw = morphisms.ExpandDims(YZW, XYZW)
expand_yzw_to_yzuw = morphisms.ExpandDims(YZW, YZUW)
expand_yzw_to_yzvw = morphisms.ExpandDims(YZW, YZVW)
expand_yuv_to_xyuv = morphisms.ExpandDims(YUV, XYUV)
expand_yuv_to_yzuv = morphisms.ExpandDims(YUV, YZUV)
expand_yuv_to_yuvw = morphisms.ExpandDims(YUV, YUVW)
expand_yuw_to_xyuw = morphisms.ExpandDims(YUW, XYUW)
expand_yuw_to_yzuw = morphisms.ExpandDims(YUW, YZUW)
expand_yuw_to_yuvw = morphisms.ExpandDims(YUW, YUVW)
expand_yvw_to_xyvw = morphisms.ExpandDims(YVW, XYVW)
expand_yvw_to_yzvw = morphisms.ExpandDims(YVW, YZVW)
expand_yvw_to_yuvw = morphisms.ExpandDims(YVW, YUVW)
expand_zuv_to_xzuv = morphisms.ExpandDims(ZUV, XZUV)
expand_zuv_to_yzuv = morphisms.ExpandDims(ZUV, YZUV)
expand_zuv_to_zuvw = morphisms.ExpandDims(ZUV, ZUVW)
expand_zuw_to_xzuw = morphisms.ExpandDims(ZUW, XZUW)
expand_zuw_to_yzuw = morphisms.ExpandDims(ZUW, YZUW)
expand_zuw_to_zuvw = morphisms.ExpandDims(ZUW, ZUVW)
expand_zvw_to_xzvw = morphisms.ExpandDims(ZVW, XZVW)
expand_zvw_to_yzvw = morphisms.ExpandDims(ZVW, YZVW)
expand_zvw_to_zuvw = morphisms.ExpandDims(ZVW, ZUVW)
expand_uvw_to_xuvw = morphisms.ExpandDims(UVW, XUVW)
expand_uvw_to_yuvw = morphisms.ExpandDims(UVW, YUVW)
expand_uvw_to_zuvw = morphisms.ExpandDims(UVW, ZUVW)

# R⁴ -> R⁵ expansions
expand_xyzu_to_xyzuv = morphisms.ExpandDims(XYZU, XYZUV)
expand_xyzu_to_xyzuw = morphisms.ExpandDims(XYZU, XYZUW)
expand_xyzv_to_xyzuv = morphisms.ExpandDims(XYZV, XYZUV)
expand_xyzv_to_xyzvw = morphisms.ExpandDims(XYZV, XYZVW)
expand_xyzw_to_xyzuw = morphisms.ExpandDims(XYZW, XYZUW)
expand_xyzw_to_xyzvw = morphisms.ExpandDims(XYZW, XYZVW)
expand_xyuv_to_xyzuv = morphisms.ExpandDims(XYUV, XYZUV)
expand_xyuv_to_xyuvw = morphisms.ExpandDims(XYUV, XYUVW)
expand_xyuw_to_xyzuw = morphisms.ExpandDims(XYUW, XYZUW)
expand_xyuw_to_xyuvw = morphisms.ExpandDims(XYUW, XYUVW)
expand_xyvw_to_xyzvw = morphisms.ExpandDims(XYVW, XYZVW)
expand_xyvw_to_xyuvw = morphisms.ExpandDims(XYVW, XYUVW)
expand_xzuv_to_xyzuv = morphisms.ExpandDims(XZUV, XYZUV)
expand_xzuv_to_xzuvw = morphisms.ExpandDims(XZUV, XZUVW)
expand_xzuw_to_xyzuw = morphisms.ExpandDims(XZUW, XYZUW)
expand_xzuw_to_xzuvw = morphisms.ExpandDims(XZUW, XZUVW)
expand_xzvw_to_xyzvw = morphisms.ExpandDims(XZVW, XYZVW)
expand_xzvw_to_xzuvw = morphisms.ExpandDims(XZVW, XZUVW)
expand_xuvw_to_xyuvw = morphisms.ExpandDims(XUVW, XYUVW)
expand_xuvw_to_xzuvw = morphisms.ExpandDims(XUVW, XZUVW)
expand_yzuv_to_xyzuv = morphisms.ExpandDims(YZUV, XYZUV)
expand_yzuv_to_yzuvw = morphisms.ExpandDims(YZUV, YZUVW)
expand_yzuw_to_xyzuw = morphisms.ExpandDims(YZUW, XYZUW)
expand_yzuw_to_yzuvw = morphisms.ExpandDims(YZUW, YZUVW)
expand_yzvw_to_xyzvw = morphisms.ExpandDims(YZVW, XYZVW)
expand_yzvw_to_yzuvw = morphisms.ExpandDims(YZVW, YZUVW)
expand_yuvw_to_xyuvw = morphisms.ExpandDims(YUVW, XYUVW)
expand_yuvw_to_yzuvw = morphisms.ExpandDims(YUVW, YZUVW)
expand_zuvw_to_xzuvw = morphisms.ExpandDims(ZUVW, XZUVW)
expand_zuvw_to_yzuvw = morphisms.ExpandDims(ZUVW, YZUVW)

# R⁵ -> R⁶ expansions
expand_xyzuv_to_xyzuvw = morphisms.ExpandDims(XYZUV, XYZUVW)
expand_xyzuw_to_xyzuvw = morphisms.ExpandDims(XYZUW, XYZUVW)
expand_xyzvw_to_xyzuvw = morphisms.ExpandDims(XYZVW, XYZUVW)
expand_xyuvw_to_xyzuvw = morphisms.ExpandDims(XYUVW, XYZUVW)
expand_xzuvw_to_xyzuvw = morphisms.ExpandDims(XZUVW, XYZUVW)
expand_yzuvw_to_xyzuvw = morphisms.ExpandDims(YZUVW, XYZUVW)

# R² -> R¹ projections
project_xy_to_x = morphisms.ProjectUsingSum(XY, X)
project_xy_to_y = morphisms.ProjectUsingSum(XY, Y)
project_xz_to_x = morphisms.ProjectUsingSum(XZ, X)
project_xz_to_z = morphisms.ProjectUsingSum(XZ, Z)
project_xu_to_x = morphisms.ProjectUsingSum(XU, X)
project_xu_to_u = morphisms.ProjectUsingSum(XU, U)
project_xv_to_x = morphisms.ProjectUsingSum(XV, X)
project_xv_to_v = morphisms.ProjectUsingSum(XV, V)
project_xw_to_x = morphisms.ProjectUsingSum(XW, X)
project_xw_to_w = morphisms.ProjectUsingSum(XW, W)
project_yz_to_y = morphisms.ProjectUsingSum(YZ, Y)
project_yz_to_z = morphisms.ProjectUsingSum(YZ, Z)
project_yu_to_y = morphisms.ProjectUsingSum(YU, Y)
project_yu_to_u = morphisms.ProjectUsingSum(YU, U)
project_yv_to_y = morphisms.ProjectUsingSum(YV, Y)
project_yv_to_v = morphisms.ProjectUsingSum(YV, V)
project_yw_to_y = morphisms.ProjectUsingSum(YW, Y)
project_yw_to_w = morphisms.ProjectUsingSum(YW, W)
project_zu_to_z = morphisms.ProjectUsingSum(ZU, Z)
project_zu_to_u = morphisms.ProjectUsingSum(ZU, U)
project_zv_to_z = morphisms.ProjectUsingSum(ZV, Z)
project_zv_to_v = morphisms.ProjectUsingSum(ZV, V)
project_zw_to_z = morphisms.ProjectUsingSum(ZW, Z)
project_zw_to_w = morphisms.ProjectUsingSum(ZW, W)
project_uv_to_u = morphisms.ProjectUsingSum(UV, U)
project_uv_to_v = morphisms.ProjectUsingSum(UV, V)
project_uw_to_u = morphisms.ProjectUsingSum(UW, U)
project_uw_to_w = morphisms.ProjectUsingSum(UW, W)
project_vw_to_v = morphisms.ProjectUsingSum(VW, V)
project_vw_to_w = morphisms.ProjectUsingSum(VW, W)

# R³ -> R² projections
project_xyz_to_xy = morphisms.ProjectUsingSum(XYZ, XY)
project_xyz_to_xz = morphisms.ProjectUsingSum(XYZ, XZ)
project_xyz_to_yz = morphisms.ProjectUsingSum(XYZ, YZ)
project_xyu_to_xy = morphisms.ProjectUsingSum(XYU, XY)
project_xyu_to_xu = morphisms.ProjectUsingSum(XYU, XU)
project_xyu_to_yu = morphisms.ProjectUsingSum(XYU, YU)
project_xyv_to_xy = morphisms.ProjectUsingSum(XYV, XY)
project_xyv_to_xv = morphisms.ProjectUsingSum(XYV, XV)
project_xyv_to_yv = morphisms.ProjectUsingSum(XYV, YV)
project_xyw_to_xy = morphisms.ProjectUsingSum(XYW, XY)
project_xyw_to_xw = morphisms.ProjectUsingSum(XYW, XW)
project_xyw_to_yw = morphisms.ProjectUsingSum(XYW, YW)
project_xzu_to_xz = morphisms.ProjectUsingSum(XZU, XZ)
project_xzu_to_xu = morphisms.ProjectUsingSum(XZU, XU)
project_xzu_to_zu = morphisms.ProjectUsingSum(XZU, ZU)
project_xzv_to_xz = morphisms.ProjectUsingSum(XZV, XZ)
project_xzv_to_xv = morphisms.ProjectUsingSum(XZV, XV)
project_xzv_to_zv = morphisms.ProjectUsingSum(XZV, ZV)
project_xzw_to_xz = morphisms.ProjectUsingSum(XZW, XZ)
project_xzw_to_xw = morphisms.ProjectUsingSum(XZW, XW)
project_xzw_to_zw = morphisms.ProjectUsingSum(XZW, ZW)
project_xuv_to_xu = morphisms.ProjectUsingSum(XUV, XU)
project_xuv_to_xv = morphisms.ProjectUsingSum(XUV, XV)
project_xuv_to_uv = morphisms.ProjectUsingSum(XUV, UV)
project_xuw_to_xu = morphisms.ProjectUsingSum(XUW, XU)
project_xuw_to_xw = morphisms.ProjectUsingSum(XUW, XW)
project_xuw_to_uw = morphisms.ProjectUsingSum(XUW, UW)
project_xvw_to_xv = morphisms.ProjectUsingSum(XVW, XV)
project_xvw_to_xw = morphisms.ProjectUsingSum(XVW, XW)
project_xvw_to_vw = morphisms.ProjectUsingSum(XVW, VW)
project_yzu_to_yz = morphisms.ProjectUsingSum(YZU, YZ)
project_yzu_to_yu = morphisms.ProjectUsingSum(YZU, YU)
project_yzu_to_zu = morphisms.ProjectUsingSum(YZU, ZU)
project_yzv_to_yz = morphisms.ProjectUsingSum(YZV, YZ)
project_yzv_to_yv = morphisms.ProjectUsingSum(YZV, YV)
project_yzv_to_zv = morphisms.ProjectUsingSum(YZV, ZV)
project_yzw_to_yz = morphisms.ProjectUsingSum(YZW, YZ)
project_yzw_to_yw = morphisms.ProjectUsingSum(YZW, YW)
project_yzw_to_zw = morphisms.ProjectUsingSum(YZW, ZW)
project_yuv_to_yu = morphisms.ProjectUsingSum(YUV, YU)
project_yuv_to_yv = morphisms.ProjectUsingSum(YUV, YV)
project_yuv_to_uv = morphisms.ProjectUsingSum(YUV, UV)
project_yuw_to_yu = morphisms.ProjectUsingSum(YUW, YU)
project_yuw_to_yw = morphisms.ProjectUsingSum(YUW, YW)
project_yuw_to_uw = morphisms.ProjectUsingSum(YUW, UW)
project_yvw_to_yv = morphisms.ProjectUsingSum(YVW, YV)
project_yvw_to_yw = morphisms.ProjectUsingSum(YVW, YW)
project_yvw_to_vw = morphisms.ProjectUsingSum(YVW, VW)
project_zuv_to_zu = morphisms.ProjectUsingSum(ZUV, ZU)
project_zuv_to_zv = morphisms.ProjectUsingSum(ZUV, ZV)
project_zuv_to_uv = morphisms.ProjectUsingSum(ZUV, UV)
project_zuw_to_zu = morphisms.ProjectUsingSum(ZUW, ZU)
project_zuw_to_zw = morphisms.ProjectUsingSum(ZUW, ZW)
project_zuw_to_uw = morphisms.ProjectUsingSum(ZUW, UW)
project_zvw_to_zv = morphisms.ProjectUsingSum(ZVW, ZV)
project_zvw_to_zw = morphisms.ProjectUsingSum(ZVW, ZW)
project_zvw_to_vw = morphisms.ProjectUsingSum(ZVW, VW)
project_uvw_to_uv = morphisms.ProjectUsingSum(UVW, UV)
project_uvw_to_uw = morphisms.ProjectUsingSum(UVW, UW)
project_uvw_to_vw = morphisms.ProjectUsingSum(UVW, VW)

# R⁴ -> R³ projections
project_xyzu_to_xyz = morphisms.ProjectUsingSum(XYZU, XYZ)
project_xyzu_to_xyu = morphisms.ProjectUsingSum(XYZU, XYU)
project_xyzu_to_xzu = morphisms.ProjectUsingSum(XYZU, XZU)
project_xyzu_to_yzu = morphisms.ProjectUsingSum(XYZU, YZU)
project_xyzv_to_xyz = morphisms.ProjectUsingSum(XYZV, XYZ)
project_xyzv_to_xyv = morphisms.ProjectUsingSum(XYZV, XYV)
project_xyzv_to_xzv = morphisms.ProjectUsingSum(XYZV, XZV)
project_xyzv_to_yzv = morphisms.ProjectUsingSum(XYZV, YZV)
project_xyzw_to_xyz = morphisms.ProjectUsingSum(XYZW, XYZ)
project_xyzw_to_xyw = morphisms.ProjectUsingSum(XYZW, XYW)
project_xyzw_to_xzw = morphisms.ProjectUsingSum(XYZW, XZW)
project_xyzw_to_yzw = morphisms.ProjectUsingSum(XYZW, YZW)
project_xyuv_to_xyu = morphisms.ProjectUsingSum(XYUV, XYU)
project_xyuv_to_xuv = morphisms.ProjectUsingSum(XYUV, XUV)
project_xyuv_to_yuv = morphisms.ProjectUsingSum(XYUV, YUV)
project_xyuw_to_xyu = morphisms.ProjectUsingSum(XYUW, XYU)
project_xyuw_to_xuw = morphisms.ProjectUsingSum(XYUW, XUW)
project_xyuw_to_yuw = morphisms.ProjectUsingSum(XYUW, YUW)
project_xyvw_to_xyv = morphisms.ProjectUsingSum(XYVW, XYV)
project_xyvw_to_xvw = morphisms.ProjectUsingSum(XYVW, XVW)
project_xyvw_to_yvw = morphisms.ProjectUsingSum(XYVW, YVW)
project_xzuv_to_xzu = morphisms.ProjectUsingSum(XZUV, XZU)
project_xzuv_to_xuv = morphisms.ProjectUsingSum(XZUV, XUV)
project_xzuv_to_zuv = morphisms.ProjectUsingSum(XZUV, ZUV)
project_xzuw_to_xzu = morphisms.ProjectUsingSum(XZUW, XZU)
project_xzuw_to_xuw = morphisms.ProjectUsingSum(XZUW, XUW)
project_xzuw_to_zuw = morphisms.ProjectUsingSum(XZUW, ZUW)
project_xzvw_to_xzv = morphisms.ProjectUsingSum(XZVW, XZV)
project_xzvw_to_xvw = morphisms.ProjectUsingSum(XZVW, XVW)
project_xzvw_to_zvw = morphisms.ProjectUsingSum(XZVW, ZVW)
project_xuvw_to_xuv = morphisms.ProjectUsingSum(XUVW, XUV)
project_xuvw_to_xuw = morphisms.ProjectUsingSum(XUVW, XUW)
project_xuvw_to_uvw = morphisms.ProjectUsingSum(XUVW, UVW)
project_yzuv_to_yzu = morphisms.ProjectUsingSum(YZUV, YZU)
project_yzuv_to_yuv = morphisms.ProjectUsingSum(YZUV, YUV)
project_yzuv_to_zuv = morphisms.ProjectUsingSum(YZUV, ZUV)
project_yzuw_to_yzu = morphisms.ProjectUsingSum(YZUW, YZU)
project_yzuw_to_yuw = morphisms.ProjectUsingSum(YZUW, YUW)
project_yzuw_to_zuw = morphisms.ProjectUsingSum(YZUW, ZUW)
project_yzvw_to_yzv = morphisms.ProjectUsingSum(YZVW, YZV)
project_yzvw_to_yvw = morphisms.ProjectUsingSum(YZVW, YVW)
project_yzvw_to_zvw = morphisms.ProjectUsingSum(YZVW, ZVW)
project_yuvw_to_yuv = morphisms.ProjectUsingSum(YUVW, YUV)
project_yuvw_to_yuw = morphisms.ProjectUsingSum(YUVW, YUW)
project_yuvw_to_uvw = morphisms.ProjectUsingSum(YUVW, UVW)
project_zuvw_to_zuv = morphisms.ProjectUsingSum(ZUVW, ZUV)
project_zuvw_to_zuw = morphisms.ProjectUsingSum(ZUVW, ZUW)
project_zuvw_to_uvw = morphisms.ProjectUsingSum(ZUVW, UVW)

# R⁵ -> R⁴ projections
project_xyzuv_to_xyzu = morphisms.ProjectUsingSum(XYZUV, XYZU)
project_xyzuv_to_xyzv = morphisms.ProjectUsingSum(XYZUV, XYZV)
project_xyzuv_to_xyuv = morphisms.ProjectUsingSum(XYZUV, XYUV)
project_xyzuv_to_xzuv = morphisms.ProjectUsingSum(XYZUV, XZUV)
project_xyzuv_to_yzuv = morphisms.ProjectUsingSum(XYZUV, YZUV)
project_xyzuw_to_xyzu = morphisms.ProjectUsingSum(XYZUW, XYZU)
project_xyzuw_to_xyzw = morphisms.ProjectUsingSum(XYZUW, XYZW)
project_xyzuw_to_xyuw = morphisms.ProjectUsingSum(XYZUW, XYUW)
project_xyzuw_to_xzuw = morphisms.ProjectUsingSum(XYZUW, XZUW)
project_xyzuw_to_yzuw = morphisms.ProjectUsingSum(XYZUW, YZUW)
project_xyzvw_to_xyzv = morphisms.ProjectUsingSum(XYZVW, XYZV)
project_xyzvw_to_xyzw = morphisms.ProjectUsingSum(XYZVW, XYZW)
project_xyzvw_to_xyvw = morphisms.ProjectUsingSum(XYZVW, XYVW)
project_xyzvw_to_xzvw = morphisms.ProjectUsingSum(XYZVW, XZVW)
project_xyzvw_to_yzvw = morphisms.ProjectUsingSum(XYZVW, YZVW)
project_xyuvw_to_xyuv = morphisms.ProjectUsingSum(XYUVW, XYUV)
project_xyuvw_to_xyuw = morphisms.ProjectUsingSum(XYUVW, XYUW)
project_xyuvw_to_xyvw = morphisms.ProjectUsingSum(XYUVW, XYVW)
project_xyuvw_to_xuvw = morphisms.ProjectUsingSum(XYUVW, XUVW)
project_xyuvw_to_yuvw = morphisms.ProjectUsingSum(XYUVW, YUVW)
project_xzuvw_to_xzuv = morphisms.ProjectUsingSum(XZUVW, XZUV)
project_xzuvw_to_xzuw = morphisms.ProjectUsingSum(XZUVW, XZUW)
project_xzuvw_to_xzvw = morphisms.ProjectUsingSum(XZUVW, XZVW)
project_xzuvw_to_xuvw = morphisms.ProjectUsingSum(XZUVW, XUVW)
project_xzuvw_to_zuvw = morphisms.ProjectUsingSum(XZUVW, ZUVW)
project_yzuvw_to_yzuv = morphisms.ProjectUsingSum(YZUVW, YZUV)
project_yzuvw_to_yzuw = morphisms.ProjectUsingSum(YZUVW, YZUW)
project_yzuvw_to_yzvw = morphisms.ProjectUsingSum(YZUVW, YZVW)
project_yzuvw_to_yuvw = morphisms.ProjectUsingSum(YZUVW, YUVW)
project_yzuvw_to_zuvw = morphisms.ProjectUsingSum(YZUVW, ZUVW)

# R⁶ -> R⁵ projections
project_xyzuvw_to_xyzuv = morphisms.ProjectUsingSum(XYZUVW, XYZUV)
project_xyzuvw_to_xyzuw = morphisms.ProjectUsingSum(XYZUVW, XYZUW)
project_xyzuvw_to_xyzvw = morphisms.ProjectUsingSum(XYZUVW, XYZVW)
project_xyzuvw_to_xyuvw = morphisms.ProjectUsingSum(XYZUVW, XYUVW)
project_xyzuvw_to_xzuvw = morphisms.ProjectUsingSum(XYZUVW, XZUVW)
project_xyzuvw_to_yzuvw = morphisms.ProjectUsingSum(XYZUVW, YZUVW)
