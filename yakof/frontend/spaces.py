"""
Bases, Spaces and Morphisms
===========================

This module provides type definitions, tensor spaces, and morphisms
up to R⁶, using the canonical dimensions X, Y, Z, U, V, and W.

Each basis defines its axes as a set of integers following the canonical ordering:
X=0, Y=1, Z=2, U=3, V=4, W=5. These are used by the morphism classes to
automatically compute the correct axes for expansions and projections.

Type Definitions and Spaces:
--------------------------
- R¹: X, Y, Z, U, V, W (single axes)
- R²: XY, XZ, YZ, ... (15 pairs of axes)
- R³: XYZ, XYU, ... (20 combinations)
- R⁴: XYZU, ... (15 combinations)
- R⁵: XYZUV, ... (6 combinations)
- R⁶: XYZUVW (1 combination)

This file defines both the basis classes (e.g., X, Y) and corresponding tensor
spaces (e.g., x, y) for each of these dimensions.

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
    >>> from yakof.frontend import abstract, spaces, morphisms
    >>> space_x = spaces.x  # Pre-defined space
    >>> x = space_x.placeholder("x")
    >>> expand = morphisms.ExpandDims(spaces.X, spaces.XY)
    >>> xy = expand(x)
"""

from . import abstract, morphisms

# Get canonical axes
x_axis, y_axis, z_axis, u_axis, v_axis, w_axis = morphisms.generate_canonical_axes(6)


# R¹ bases
class X:
    """Tensor basis along the X axis."""

    axes = (x_axis,)


class Y:
    """Tensor basis along the Y axis."""

    axes = (y_axis,)


class Z:
    """Tensor basis along the Z axis."""

    axes = (z_axis,)


class U:
    """Tensor basis along the U axis."""

    axes = (u_axis,)


class V:
    """Tensor basis along the V axis."""

    axes = (v_axis,)


class W:
    """Tensor basis along the W axis."""

    axes = (w_axis,)


# R² bases
class XY:
    """Tensor basis along X and Y axes."""

    axes = (x_axis, y_axis)


class XZ:
    """Tensor basis along X and Z axes."""

    axes = (x_axis, z_axis)


class XU:
    """Tensor basis along X and U axes."""

    axes = (x_axis, u_axis)


class XV:
    """Tensor basis along X and V axes."""

    axes = (x_axis, v_axis)


class XW:
    """Tensor basis along X and W axes."""

    axes = (x_axis, w_axis)


class YZ:
    """Tensor basis along Y and Z axes."""

    axes = (y_axis, z_axis)


class YU:
    """Tensor basis along Y and U axes."""

    axes = (y_axis, u_axis)


class YV:
    """Tensor basis along Y and V axes."""

    axes = (y_axis, v_axis)


class YW:
    """Tensor basis along Y and W axes."""

    axes = (y_axis, w_axis)


class ZU:
    """Tensor basis along Z and U axes."""

    axes = (z_axis, u_axis)


class ZV:
    """Tensor basis along Z and V axes."""

    axes = (z_axis, v_axis)


class ZW:
    """Tensor basis along Z and W axes."""

    axes = (z_axis, w_axis)


class UV:
    """Tensor basis along U and V axes."""

    axes = (u_axis, v_axis)


class UW:
    """Tensor basis along U and W axes."""

    axes = (u_axis, w_axis)


class VW:
    """Tensor basis along V and W axes."""

    axes = (v_axis, w_axis)


# R³ bases
class XYZ:
    """Tensor basis along X, Y and Z axes."""

    axes = (x_axis, y_axis, z_axis)


class XYU:
    """Tensor basis along X, Y and U axes."""

    axes = (x_axis, y_axis, u_axis)


class XYV:
    """Tensor basis along X, Y and V axes."""

    axes = (x_axis, y_axis, v_axis)


class XYW:
    """Tensor basis along X, Y and W axes."""

    axes = (x_axis, y_axis, w_axis)


class XZU:
    """Tensor basis along X, Z and U axes."""

    axes = (x_axis, z_axis, u_axis)


class XZV:
    """Tensor basis along X, Z and V axes."""

    axes = (x_axis, z_axis, v_axis)


class XZW:
    """Tensor basis along X, Z and W axes."""

    axes = (x_axis, z_axis, w_axis)


class XUV:
    """Tensor basis along X, U and V axes."""

    axes = (x_axis, u_axis, v_axis)


class XUW:
    """Tensor basis along X, U and W axes."""

    axes = (x_axis, u_axis, w_axis)


class XVW:
    """Tensor basis along X, V and W axes."""

    axes = (x_axis, v_axis, w_axis)


class YZU:
    """Tensor basis along Y, Z and U axes."""

    axes = (y_axis, z_axis, u_axis)


class YZV:
    """Tensor basis along Y, Z and V axes."""

    axes = (y_axis, z_axis, v_axis)


class YZW:
    """Tensor basis along Y, Z and W axes."""

    axes = (y_axis, z_axis, w_axis)


class YUV:
    """Tensor basis along Y, U and V axes."""

    axes = (y_axis, u_axis, v_axis)


class YUW:
    """Tensor basis along Y, U and W axes."""

    axes = (y_axis, u_axis, w_axis)


class YVW:
    """Tensor basis along Y, V and W axes."""

    axes = (y_axis, v_axis, w_axis)


class ZUV:
    """Tensor basis along Z, U and V axes."""

    axes = (z_axis, u_axis, v_axis)


class ZUW:
    """Tensor basis along Z, U and W axes."""

    axes = (z_axis, u_axis, w_axis)


class ZVW:
    """Tensor basis along Z, V and W axes."""

    axes = (z_axis, v_axis, w_axis)


class UVW:
    """Tensor basis along U, V and W axes."""

    axes = (u_axis, v_axis, w_axis)


# R⁴ bases
class XYZU:
    """Tensor basis along X, Y, Z and U axes."""

    axes = (x_axis, y_axis, z_axis, u_axis)


class XYZV:
    """Tensor basis along X, Y, Z and V axes."""

    axes = (x_axis, y_axis, z_axis, v_axis)


class XYZW:
    """Tensor basis along X, Y, Z and W axes."""

    axes = (x_axis, y_axis, z_axis, w_axis)


class XYUV:
    """Tensor basis along X, Y, U and V axes."""

    axes = (x_axis, y_axis, u_axis, v_axis)


class XYUW:
    """Tensor basis along X, Y, U and W axes."""

    axes = (x_axis, y_axis, u_axis, w_axis)


class XYVW:
    """Tensor basis along X, Y, V and W axes."""

    axes = (x_axis, y_axis, v_axis, w_axis)


class XZUV:
    """Tensor basis along X, Z, U and V axes."""

    axes = (x_axis, z_axis, u_axis, v_axis)


class XZUW:
    """Tensor basis along X, Z, U and W axes."""

    axes = (x_axis, z_axis, u_axis, w_axis)


class XZVW:
    """Tensor basis along X, Z, V and W axes."""

    axes = (x_axis, z_axis, v_axis, w_axis)


class XUVW:
    """Tensor basis along X, U, V and W axes."""

    axes = (x_axis, u_axis, v_axis, w_axis)


class YZUV:
    """Tensor basis along Y, Z, U and V axes."""

    axes = (y_axis, z_axis, u_axis, v_axis)


class YZUW:
    """Tensor basis along Y, Z, U and W axes."""

    axes = (y_axis, z_axis, u_axis, w_axis)


class YZVW:
    """Tensor basis along Y, Z, V and W axes."""

    axes = (y_axis, z_axis, v_axis, w_axis)


class YUVW:
    """Tensor basis along Y, U, V and W axes."""

    axes = (y_axis, u_axis, v_axis, w_axis)


class ZUVW:
    """Tensor basis along Z, U, V and W axes."""

    axes = (z_axis, u_axis, v_axis, w_axis)


# R⁵ bases
class XYZUV:
    """Tensor basis along X, Y, Z, U and V axes."""

    axes = (x_axis, y_axis, z_axis, u_axis, v_axis)


class XYZUW:
    """Tensor basis along X, Y, Z, U and W axes."""

    axes = (x_axis, y_axis, z_axis, u_axis, w_axis)


class XYZVW:
    """Tensor basis along X, Y, Z, V and W axes."""

    axes = (x_axis, y_axis, z_axis, v_axis, w_axis)


class XYUVW:
    """Tensor basis along X, Y, U, V and W axes."""

    axes = (x_axis, y_axis, u_axis, v_axis, w_axis)


class XZUVW:
    """Tensor basis along X, Z, U, V and W axes."""

    axes = (x_axis, z_axis, u_axis, v_axis, w_axis)


class YZUVW:
    """Tensor basis along Y, Z, U, V and W axes."""

    axes = (y_axis, z_axis, u_axis, v_axis, w_axis)


# R⁶ basis
class XYZUVW:
    """Tensor basis along X, Y, Z, U, V and W axes."""

    axes = (x_axis, y_axis, z_axis, u_axis, v_axis, w_axis)


# R¹ tensor spaces
x = abstract.TensorSpace(X())
y = abstract.TensorSpace(Y())
z = abstract.TensorSpace(Z())
u = abstract.TensorSpace(U())
v = abstract.TensorSpace(V())
w = abstract.TensorSpace(W())

# R² tensor spaces
xy = abstract.TensorSpace(XY())
xz = abstract.TensorSpace(XZ())
xu = abstract.TensorSpace(XU())
xv = abstract.TensorSpace(XV())
xw = abstract.TensorSpace(XW())
yz = abstract.TensorSpace(YZ())
yu = abstract.TensorSpace(YU())
yv = abstract.TensorSpace(YV())
yw = abstract.TensorSpace(YW())
zu = abstract.TensorSpace(ZU())
zv = abstract.TensorSpace(ZV())
zw = abstract.TensorSpace(ZW())
uv = abstract.TensorSpace(UV())
uw = abstract.TensorSpace(UW())
vw = abstract.TensorSpace(VW())

# R³ tensor spaces
xyz = abstract.TensorSpace(XYZ())
xyu = abstract.TensorSpace(XYU())
xyv = abstract.TensorSpace(XYV())
xyw = abstract.TensorSpace(XYW())
xzu = abstract.TensorSpace(XZU())
xzv = abstract.TensorSpace(XZV())
xzw = abstract.TensorSpace(XZW())
xuv = abstract.TensorSpace(XUV())
xuw = abstract.TensorSpace(XUW())
xvw = abstract.TensorSpace(XVW())
yzu = abstract.TensorSpace(YZU())
yzv = abstract.TensorSpace(YZV())
yzw = abstract.TensorSpace(YZW())
yuv = abstract.TensorSpace(YUV())
yuw = abstract.TensorSpace(YUW())
yvw = abstract.TensorSpace(YVW())
zuv = abstract.TensorSpace(ZUV())
zuw = abstract.TensorSpace(ZUW())
zvw = abstract.TensorSpace(ZVW())
uvw = abstract.TensorSpace(UVW())

# R⁴ tensor spaces
xyzu = abstract.TensorSpace(XYZU())
xyzv = abstract.TensorSpace(XYZV())
xyzw = abstract.TensorSpace(XYZW())
xyuv = abstract.TensorSpace(XYUV())
xyuw = abstract.TensorSpace(XYUW())
xyvw = abstract.TensorSpace(XYVW())
xzuv = abstract.TensorSpace(XZUV())
xzuw = abstract.TensorSpace(XZUW())
xzvw = abstract.TensorSpace(XZVW())
xuvw = abstract.TensorSpace(XUVW())
yzuv = abstract.TensorSpace(YZUV())
yzuw = abstract.TensorSpace(YZUW())
yzvw = abstract.TensorSpace(YZVW())
yuvw = abstract.TensorSpace(YUVW())
zuvw = abstract.TensorSpace(ZUVW())

# R⁵ tensor spaces
xyzuv = abstract.TensorSpace(XYZUV())
xyzuw = abstract.TensorSpace(XYZUW())
xyzvw = abstract.TensorSpace(XYZVW())
xyuvw = abstract.TensorSpace(XYUVW())
xzuvw = abstract.TensorSpace(XZUVW())
yzuvw = abstract.TensorSpace(YZUVW())

# R⁶ tensor space
xyzuvw = abstract.TensorSpace(XYZUVW())


# R¹ -> R² expansions

expand_x_to_xy = morphisms.ExpandDims(x, xy)
expand_x_to_xz = morphisms.ExpandDims(x, xz)
expand_x_to_xu = morphisms.ExpandDims(x, xu)
expand_x_to_xv = morphisms.ExpandDims(x, xv)
expand_x_to_xw = morphisms.ExpandDims(x, xw)
expand_y_to_xy = morphisms.ExpandDims(y, xy)
expand_y_to_yz = morphisms.ExpandDims(y, yz)
expand_y_to_yu = morphisms.ExpandDims(y, yu)
expand_y_to_yv = morphisms.ExpandDims(y, yv)
expand_y_to_yw = morphisms.ExpandDims(y, yw)
expand_z_to_xz = morphisms.ExpandDims(z, xz)
expand_z_to_yz = morphisms.ExpandDims(z, yz)
expand_z_to_zu = morphisms.ExpandDims(z, zu)
expand_z_to_zv = morphisms.ExpandDims(z, zv)
expand_z_to_zw = morphisms.ExpandDims(z, zw)
expand_u_to_xu = morphisms.ExpandDims(u, xu)
expand_u_to_yu = morphisms.ExpandDims(u, yu)
expand_u_to_zu = morphisms.ExpandDims(u, zu)
expand_u_to_uv = morphisms.ExpandDims(u, uv)
expand_u_to_uw = morphisms.ExpandDims(u, uw)
expand_v_to_xv = morphisms.ExpandDims(v, xv)
expand_v_to_yv = morphisms.ExpandDims(v, yv)
expand_v_to_zv = morphisms.ExpandDims(v, zv)
expand_v_to_uv = morphisms.ExpandDims(v, uv)
expand_v_to_vw = morphisms.ExpandDims(v, vw)
expand_w_to_xw = morphisms.ExpandDims(w, xw)
expand_w_to_yw = morphisms.ExpandDims(w, yw)
expand_w_to_zw = morphisms.ExpandDims(w, zw)
expand_w_to_uw = morphisms.ExpandDims(w, uw)
expand_w_to_vw = morphisms.ExpandDims(w, vw)

# R² -> R³ expansions
expand_xy_to_xyz = morphisms.ExpandDims(xy, xyz)
expand_xy_to_xyu = morphisms.ExpandDims(xy, xyu)
expand_xy_to_xyv = morphisms.ExpandDims(xy, xyv)
expand_xy_to_xyw = morphisms.ExpandDims(xy, xyw)
expand_xz_to_xyz = morphisms.ExpandDims(xz, xyz)
expand_xz_to_xzu = morphisms.ExpandDims(xz, xzu)
expand_xz_to_xzv = morphisms.ExpandDims(xz, xzv)
expand_xz_to_xzw = morphisms.ExpandDims(xz, xzw)
expand_xu_to_xyu = morphisms.ExpandDims(xu, xyu)
expand_xu_to_xzu = morphisms.ExpandDims(xu, xzu)
expand_xu_to_xuv = morphisms.ExpandDims(xu, xuv)
expand_xu_to_xuw = morphisms.ExpandDims(xu, xuw)
expand_xv_to_xyv = morphisms.ExpandDims(xv, xyv)
expand_xv_to_xzv = morphisms.ExpandDims(xv, xzv)
expand_xv_to_xuv = morphisms.ExpandDims(xv, xuv)
expand_xv_to_xvw = morphisms.ExpandDims(xv, xvw)
expand_xw_to_xyw = morphisms.ExpandDims(xw, xyw)
expand_xw_to_xzw = morphisms.ExpandDims(xw, xzw)
expand_xw_to_xuw = morphisms.ExpandDims(xw, xuw)
expand_xw_to_xvw = morphisms.ExpandDims(xw, xvw)
expand_yz_to_xyz = morphisms.ExpandDims(yz, xyz)
expand_yz_to_yzu = morphisms.ExpandDims(yz, yzu)
expand_yz_to_yzv = morphisms.ExpandDims(yz, yzv)
expand_yz_to_yzw = morphisms.ExpandDims(yz, yzw)
expand_yu_to_xyu = morphisms.ExpandDims(yu, xyu)
expand_yu_to_yzu = morphisms.ExpandDims(yu, yzu)
expand_yu_to_yuv = morphisms.ExpandDims(yu, yuv)
expand_yu_to_yuw = morphisms.ExpandDims(yu, yuw)
expand_yv_to_xyv = morphisms.ExpandDims(yv, xyv)
expand_yv_to_yzv = morphisms.ExpandDims(yv, yzv)
expand_yv_to_yuv = morphisms.ExpandDims(yv, yuv)
expand_yv_to_yvw = morphisms.ExpandDims(yv, yvw)
expand_yw_to_xyw = morphisms.ExpandDims(yw, xyw)
expand_yw_to_yzw = morphisms.ExpandDims(yw, yzw)
expand_yw_to_yuw = morphisms.ExpandDims(yw, yuw)
expand_yw_to_yvw = morphisms.ExpandDims(yw, yvw)
expand_zu_to_xzu = morphisms.ExpandDims(zu, xzu)
expand_zu_to_yzu = morphisms.ExpandDims(zu, yzu)
expand_zu_to_zuv = morphisms.ExpandDims(zu, zuv)
expand_zu_to_zuw = morphisms.ExpandDims(zu, zuw)
expand_zv_to_xzv = morphisms.ExpandDims(zv, xzv)
expand_zv_to_yzv = morphisms.ExpandDims(zv, yzv)
expand_zv_to_zuv = morphisms.ExpandDims(zv, zuv)
expand_zv_to_zvw = morphisms.ExpandDims(zv, zvw)
expand_zw_to_xzw = morphisms.ExpandDims(zw, xzw)
expand_zw_to_yzw = morphisms.ExpandDims(zw, yzw)
expand_zw_to_zuw = morphisms.ExpandDims(zw, zuw)
expand_zw_to_zvw = morphisms.ExpandDims(zw, zvw)
expand_uv_to_xuv = morphisms.ExpandDims(uv, xuv)
expand_uv_to_yuv = morphisms.ExpandDims(uv, yuv)
expand_uv_to_zuv = morphisms.ExpandDims(uv, zuv)
expand_uv_to_uvw = morphisms.ExpandDims(uv, uvw)
expand_uw_to_xuw = morphisms.ExpandDims(uw, xuw)
expand_uw_to_yuw = morphisms.ExpandDims(uw, yuw)
expand_uw_to_zuw = morphisms.ExpandDims(uw, zuw)
expand_uw_to_uvw = morphisms.ExpandDims(uw, uvw)
expand_vw_to_xvw = morphisms.ExpandDims(vw, xvw)
expand_vw_to_yvw = morphisms.ExpandDims(vw, yvw)
expand_vw_to_zvw = morphisms.ExpandDims(vw, zvw)
expand_vw_to_uvw = morphisms.ExpandDims(vw, uvw)

# R³ -> R⁴ expansions
expand_xyz_to_xyzu = morphisms.ExpandDims(xyz, xyzu)
expand_xyz_to_xyzv = morphisms.ExpandDims(xyz, xyzv)
expand_xyz_to_xyzw = morphisms.ExpandDims(xyz, xyzw)
expand_xyu_to_xyzu = morphisms.ExpandDims(xyu, xyzu)
expand_xyu_to_xyuv = morphisms.ExpandDims(xyu, xyuv)
expand_xyu_to_xyuw = morphisms.ExpandDims(xyu, xyuw)
expand_xyv_to_xyzv = morphisms.ExpandDims(xyv, xyzv)
expand_xyv_to_xyuv = morphisms.ExpandDims(xyv, xyuv)
expand_xyv_to_xyvw = morphisms.ExpandDims(xyv, xyvw)
expand_xyw_to_xyzw = morphisms.ExpandDims(xyw, xyzw)
expand_xyw_to_xyuw = morphisms.ExpandDims(xyw, xyuw)
expand_xyw_to_xyvw = morphisms.ExpandDims(xyw, xyvw)
expand_xzu_to_xyzu = morphisms.ExpandDims(xzu, xyzu)
expand_xzu_to_xzuv = morphisms.ExpandDims(xzu, xzuv)
expand_xzu_to_xzuw = morphisms.ExpandDims(xzu, xzuw)
expand_xzv_to_xyzv = morphisms.ExpandDims(xzv, xyzv)
expand_xzv_to_xzuv = morphisms.ExpandDims(xzv, xzuv)
expand_xzv_to_xzvw = morphisms.ExpandDims(xzv, xzvw)
expand_xzw_to_xyzw = morphisms.ExpandDims(xzw, xyzw)
expand_xzw_to_xzuw = morphisms.ExpandDims(xzw, xzuw)
expand_xzw_to_xzvw = morphisms.ExpandDims(xzw, xzvw)
expand_xuv_to_xyuv = morphisms.ExpandDims(xuv, xyuv)
expand_xuv_to_xzuv = morphisms.ExpandDims(xuv, xzuv)
expand_xuv_to_xuvw = morphisms.ExpandDims(xuv, xuvw)
expand_xuw_to_xyuw = morphisms.ExpandDims(xuw, xyuw)
expand_xuw_to_xzuw = morphisms.ExpandDims(xuw, xzuw)
expand_xuw_to_xuvw = morphisms.ExpandDims(xuw, xuvw)
expand_xvw_to_xyvw = morphisms.ExpandDims(xvw, xyvw)
expand_xvw_to_xzvw = morphisms.ExpandDims(xvw, xzvw)
expand_xvw_to_xuvw = morphisms.ExpandDims(xvw, xuvw)
expand_yzu_to_xyzu = morphisms.ExpandDims(yzu, xyzu)
expand_yzu_to_yzuv = morphisms.ExpandDims(yzu, yzuv)
expand_yzu_to_yzuw = morphisms.ExpandDims(yzu, yzuw)
expand_yzv_to_xyzv = morphisms.ExpandDims(yzv, xyzv)
expand_yzv_to_yzuv = morphisms.ExpandDims(yzv, yzuv)
expand_yzv_to_yzvw = morphisms.ExpandDims(yzv, yzvw)
expand_yzw_to_xyzw = morphisms.ExpandDims(yzw, xyzw)
expand_yzw_to_yzuw = morphisms.ExpandDims(yzw, yzuw)
expand_yzw_to_yzvw = morphisms.ExpandDims(yzw, yzvw)
expand_yuv_to_xyuv = morphisms.ExpandDims(yuv, xyuv)
expand_yuv_to_yzuv = morphisms.ExpandDims(yuv, yzuv)
expand_yuv_to_yuvw = morphisms.ExpandDims(yuv, yuvw)
expand_yuw_to_xyuw = morphisms.ExpandDims(yuw, xyuw)
expand_yuw_to_yzuw = morphisms.ExpandDims(yuw, yzuw)
expand_yuw_to_yuvw = morphisms.ExpandDims(yuw, yuvw)
expand_yvw_to_xyvw = morphisms.ExpandDims(yvw, xyvw)
expand_yvw_to_yzvw = morphisms.ExpandDims(yvw, yzvw)
expand_yvw_to_yuvw = morphisms.ExpandDims(yvw, yuvw)
expand_zuv_to_xzuv = morphisms.ExpandDims(zuv, xzuv)
expand_zuv_to_yzuv = morphisms.ExpandDims(zuv, yzuv)
expand_zuv_to_zuvw = morphisms.ExpandDims(zuv, zuvw)
expand_zuw_to_xzuw = morphisms.ExpandDims(zuw, xzuw)
expand_zuw_to_yzuw = morphisms.ExpandDims(zuw, yzuw)
expand_zuw_to_zuvw = morphisms.ExpandDims(zuw, zuvw)
expand_zvw_to_xzvw = morphisms.ExpandDims(zvw, xzvw)
expand_zvw_to_yzvw = morphisms.ExpandDims(zvw, yzvw)
expand_zvw_to_zuvw = morphisms.ExpandDims(zvw, zuvw)
expand_uvw_to_xuvw = morphisms.ExpandDims(uvw, xuvw)
expand_uvw_to_yuvw = morphisms.ExpandDims(uvw, yuvw)
expand_uvw_to_zuvw = morphisms.ExpandDims(uvw, zuvw)

# R⁴ -> R⁵ expansions
expand_xyzu_to_xyzuv = morphisms.ExpandDims(xyzu, xyzuv)
expand_xyzu_to_xyzuw = morphisms.ExpandDims(xyzu, xyzuw)
expand_xyzv_to_xyzuv = morphisms.ExpandDims(xyzv, xyzuv)
expand_xyzv_to_xyzvw = morphisms.ExpandDims(xyzv, xyzvw)
expand_xyzw_to_xyzuw = morphisms.ExpandDims(xyzw, xyzuw)
expand_xyzw_to_xyzvw = morphisms.ExpandDims(xyzw, xyzvw)
expand_xyuv_to_xyzuv = morphisms.ExpandDims(xyuv, xyzuv)
expand_xyuv_to_xyuvw = morphisms.ExpandDims(xyuv, xyuvw)
expand_xyuw_to_xyzuw = morphisms.ExpandDims(xyuw, xyzuw)
expand_xyuw_to_xyuvw = morphisms.ExpandDims(xyuw, xyuvw)
expand_xyvw_to_xyzvw = morphisms.ExpandDims(xyvw, xyzvw)
expand_xyvw_to_xyuvw = morphisms.ExpandDims(xyvw, xyuvw)
expand_xzuv_to_xyzuv = morphisms.ExpandDims(xzuv, xyzuv)
expand_xzuv_to_xzuvw = morphisms.ExpandDims(xzuv, xzuvw)
expand_xzuw_to_xyzuw = morphisms.ExpandDims(xzuw, xyzuw)
expand_xzuw_to_xzuvw = morphisms.ExpandDims(xzuw, xzuvw)
expand_xzvw_to_xyzvw = morphisms.ExpandDims(xzvw, xyzvw)
expand_xzvw_to_xzuvw = morphisms.ExpandDims(xzvw, xzuvw)
expand_xuvw_to_xyuvw = morphisms.ExpandDims(xuvw, xyuvw)
expand_xuvw_to_xzuvw = morphisms.ExpandDims(xuvw, xzuvw)
expand_yzuv_to_xyzuv = morphisms.ExpandDims(yzuv, xyzuv)
expand_yzuv_to_yzuvw = morphisms.ExpandDims(yzuv, yzuvw)
expand_yzuw_to_xyzuw = morphisms.ExpandDims(yzuw, xyzuw)
expand_yzuw_to_yzuvw = morphisms.ExpandDims(yzuw, yzuvw)
expand_yzvw_to_xyzvw = morphisms.ExpandDims(yzvw, xyzvw)
expand_yzvw_to_yzuvw = morphisms.ExpandDims(yzvw, yzuvw)
expand_yuvw_to_xyuvw = morphisms.ExpandDims(yuvw, xyuvw)
expand_yuvw_to_yzuvw = morphisms.ExpandDims(yuvw, yzuvw)
expand_zuvw_to_xzuvw = morphisms.ExpandDims(zuvw, xzuvw)
expand_zuvw_to_yzuvw = morphisms.ExpandDims(zuvw, yzuvw)

# R⁵ -> R⁶ expansions
expand_xyzuv_to_xyzuvw = morphisms.ExpandDims(xyzuv, xyzuvw)
expand_xyzuw_to_xyzuvw = morphisms.ExpandDims(xyzuw, xyzuvw)
expand_xyzvw_to_xyzuvw = morphisms.ExpandDims(xyzvw, xyzuvw)
expand_xyuvw_to_xyzuvw = morphisms.ExpandDims(xyuvw, xyzuvw)
expand_xzuvw_to_xyzuvw = morphisms.ExpandDims(xzuvw, xyzuvw)
expand_yzuvw_to_xyzuvw = morphisms.ExpandDims(yzuvw, xyzuvw)

# R² -> R¹ projections
project_xy_to_x = morphisms.ProjectUsingSum(xy, x)
project_xy_to_y = morphisms.ProjectUsingSum(xy, y)
project_xz_to_x = morphisms.ProjectUsingSum(xz, x)
project_xz_to_z = morphisms.ProjectUsingSum(xz, z)
project_xu_to_x = morphisms.ProjectUsingSum(xu, x)
project_xu_to_u = morphisms.ProjectUsingSum(xu, u)
project_xv_to_x = morphisms.ProjectUsingSum(xv, x)
project_xv_to_v = morphisms.ProjectUsingSum(xv, v)
project_xw_to_x = morphisms.ProjectUsingSum(xw, x)
project_xw_to_w = morphisms.ProjectUsingSum(xw, w)
project_yz_to_y = morphisms.ProjectUsingSum(yz, y)
project_yz_to_z = morphisms.ProjectUsingSum(yz, z)
project_yu_to_y = morphisms.ProjectUsingSum(yu, y)
project_yu_to_u = morphisms.ProjectUsingSum(yu, u)
project_yv_to_y = morphisms.ProjectUsingSum(yv, y)
project_yv_to_v = morphisms.ProjectUsingSum(yv, v)
project_yw_to_y = morphisms.ProjectUsingSum(yw, y)
project_yw_to_w = morphisms.ProjectUsingSum(yw, w)
project_zu_to_z = morphisms.ProjectUsingSum(zu, z)
project_zu_to_u = morphisms.ProjectUsingSum(zu, u)
project_zv_to_z = morphisms.ProjectUsingSum(zv, z)
project_zv_to_v = morphisms.ProjectUsingSum(zv, v)
project_zw_to_z = morphisms.ProjectUsingSum(zw, z)
project_zw_to_w = morphisms.ProjectUsingSum(zw, w)
project_uv_to_u = morphisms.ProjectUsingSum(uv, u)
project_uv_to_v = morphisms.ProjectUsingSum(uv, v)
project_uw_to_u = morphisms.ProjectUsingSum(uw, u)
project_uw_to_w = morphisms.ProjectUsingSum(uw, w)
project_vw_to_v = morphisms.ProjectUsingSum(vw, v)
project_vw_to_w = morphisms.ProjectUsingSum(vw, w)

# R³ -> R² projections
project_xyz_to_xy = morphisms.ProjectUsingSum(xyz, xy)
project_xyz_to_xz = morphisms.ProjectUsingSum(xyz, xz)
project_xyz_to_yz = morphisms.ProjectUsingSum(xyz, yz)
project_xyu_to_xy = morphisms.ProjectUsingSum(xyu, xy)
project_xyu_to_xu = morphisms.ProjectUsingSum(xyu, xu)
project_xyu_to_yu = morphisms.ProjectUsingSum(xyu, yu)
project_xyv_to_xy = morphisms.ProjectUsingSum(xyv, xy)
project_xyv_to_xv = morphisms.ProjectUsingSum(xyv, xv)
project_xyv_to_yv = morphisms.ProjectUsingSum(xyv, yv)
project_xyw_to_xy = morphisms.ProjectUsingSum(xyw, xy)
project_xyw_to_xw = morphisms.ProjectUsingSum(xyw, xw)
project_xyw_to_yw = morphisms.ProjectUsingSum(xyw, yw)
project_xzu_to_xz = morphisms.ProjectUsingSum(xzu, xz)
project_xzu_to_xu = morphisms.ProjectUsingSum(xzu, xu)
project_xzu_to_zu = morphisms.ProjectUsingSum(xzu, zu)
project_xzv_to_xz = morphisms.ProjectUsingSum(xzv, xz)
project_xzv_to_xv = morphisms.ProjectUsingSum(xzv, xv)
project_xzv_to_zv = morphisms.ProjectUsingSum(xzv, zv)
project_xzw_to_xz = morphisms.ProjectUsingSum(xzw, xz)
project_xzw_to_xw = morphisms.ProjectUsingSum(xzw, xw)
project_xzw_to_zw = morphisms.ProjectUsingSum(xzw, zw)
project_xuv_to_xu = morphisms.ProjectUsingSum(xuv, xu)
project_xuv_to_xv = morphisms.ProjectUsingSum(xuv, xv)
project_xuv_to_uv = morphisms.ProjectUsingSum(xuv, uv)
project_xuw_to_xu = morphisms.ProjectUsingSum(xuw, xu)
project_xuw_to_xw = morphisms.ProjectUsingSum(xuw, xw)
project_xuw_to_uw = morphisms.ProjectUsingSum(xuw, uw)
project_xvw_to_xv = morphisms.ProjectUsingSum(xvw, xv)
project_xvw_to_xw = morphisms.ProjectUsingSum(xvw, xw)
project_xvw_to_vw = morphisms.ProjectUsingSum(xvw, vw)
project_yzu_to_yz = morphisms.ProjectUsingSum(yzu, yz)
project_yzu_to_yu = morphisms.ProjectUsingSum(yzu, yu)
project_yzu_to_zu = morphisms.ProjectUsingSum(yzu, zu)
project_yzv_to_yz = morphisms.ProjectUsingSum(yzv, yz)
project_yzv_to_yv = morphisms.ProjectUsingSum(yzv, yv)
project_yzv_to_zv = morphisms.ProjectUsingSum(yzv, zv)
project_yzw_to_yz = morphisms.ProjectUsingSum(yzw, yz)
project_yzw_to_yw = morphisms.ProjectUsingSum(yzw, yw)
project_yzw_to_zw = morphisms.ProjectUsingSum(yzw, zw)
project_yuv_to_yu = morphisms.ProjectUsingSum(yuv, yu)
project_yuv_to_yv = morphisms.ProjectUsingSum(yuv, yv)
project_yuv_to_uv = morphisms.ProjectUsingSum(yuv, uv)
project_yuw_to_yu = morphisms.ProjectUsingSum(yuw, yu)
project_yuw_to_yw = morphisms.ProjectUsingSum(yuw, yw)
project_yuw_to_uw = morphisms.ProjectUsingSum(yuw, uw)
project_yvw_to_yv = morphisms.ProjectUsingSum(yvw, yv)
project_yvw_to_yw = morphisms.ProjectUsingSum(yvw, yw)
project_yvw_to_vw = morphisms.ProjectUsingSum(yvw, vw)
project_zuv_to_zu = morphisms.ProjectUsingSum(zuv, zu)
project_zuv_to_zv = morphisms.ProjectUsingSum(zuv, zv)
project_zuv_to_uv = morphisms.ProjectUsingSum(zuv, uv)
project_zuw_to_zu = morphisms.ProjectUsingSum(zuw, zu)
project_zuw_to_zw = morphisms.ProjectUsingSum(zuw, zw)
project_zuw_to_uw = morphisms.ProjectUsingSum(zuw, uw)
project_zvw_to_zv = morphisms.ProjectUsingSum(zvw, zv)
project_zvw_to_zw = morphisms.ProjectUsingSum(zvw, zw)
project_zvw_to_vw = morphisms.ProjectUsingSum(zvw, vw)
project_uvw_to_uv = morphisms.ProjectUsingSum(uvw, uv)
project_uvw_to_uw = morphisms.ProjectUsingSum(uvw, uw)
project_uvw_to_vw = morphisms.ProjectUsingSum(uvw, vw)

# R⁴ -> R³ projections
project_xyzu_to_xyz = morphisms.ProjectUsingSum(xyzu, xyz)
project_xyzu_to_xyu = morphisms.ProjectUsingSum(xyzu, xyu)
project_xyzu_to_xzu = morphisms.ProjectUsingSum(xyzu, xzu)
project_xyzu_to_yzu = morphisms.ProjectUsingSum(xyzu, yzu)
project_xyzv_to_xyz = morphisms.ProjectUsingSum(xyzv, xyz)
project_xyzv_to_xyv = morphisms.ProjectUsingSum(xyzv, xyv)
project_xyzv_to_xzv = morphisms.ProjectUsingSum(xyzv, xzv)
project_xyzv_to_yzv = morphisms.ProjectUsingSum(xyzv, yzv)
project_xyzw_to_xyz = morphisms.ProjectUsingSum(xyzw, xyz)
project_xyzw_to_xyw = morphisms.ProjectUsingSum(xyzw, xyw)
project_xyzw_to_xzw = morphisms.ProjectUsingSum(xyzw, xzw)
project_xyzw_to_yzw = morphisms.ProjectUsingSum(xyzw, yzw)
project_xyuv_to_xyu = morphisms.ProjectUsingSum(xyuv, xyu)
project_xyuv_to_xuv = morphisms.ProjectUsingSum(xyuv, xuv)
project_xyuv_to_yuv = morphisms.ProjectUsingSum(xyuv, yuv)
project_xyuw_to_xyu = morphisms.ProjectUsingSum(xyuw, xyu)
project_xyuw_to_xuw = morphisms.ProjectUsingSum(xyuw, xuw)
project_xyuw_to_yuw = morphisms.ProjectUsingSum(xyuw, yuw)
project_xyvw_to_xyv = morphisms.ProjectUsingSum(xyvw, xyv)
project_xyvw_to_xvw = morphisms.ProjectUsingSum(xyvw, xvw)
project_xyvw_to_yvw = morphisms.ProjectUsingSum(xyvw, yvw)
project_xzuv_to_xzu = morphisms.ProjectUsingSum(xzuv, xzu)
project_xzuv_to_xuv = morphisms.ProjectUsingSum(xzuv, xuv)
project_xzuv_to_zuv = morphisms.ProjectUsingSum(xzuv, zuv)
project_xzuw_to_xzu = morphisms.ProjectUsingSum(xzuw, xzu)
project_xzuw_to_xuw = morphisms.ProjectUsingSum(xzuw, xuw)
project_xzuw_to_zuw = morphisms.ProjectUsingSum(xzuw, zuw)
project_xzvw_to_xzv = morphisms.ProjectUsingSum(xzvw, xzv)
project_xzvw_to_xvw = morphisms.ProjectUsingSum(xzvw, xvw)
project_xzvw_to_zvw = morphisms.ProjectUsingSum(xzvw, zvw)
project_xuvw_to_xuv = morphisms.ProjectUsingSum(xuvw, xuv)
project_xuvw_to_xuw = morphisms.ProjectUsingSum(xuvw, xuw)
project_xuvw_to_uvw = morphisms.ProjectUsingSum(xuvw, uvw)
project_yzuv_to_yzu = morphisms.ProjectUsingSum(yzuv, yzu)
project_yzuv_to_yuv = morphisms.ProjectUsingSum(yzuv, yuv)
project_yzuv_to_zuv = morphisms.ProjectUsingSum(yzuv, zuv)
project_yzuw_to_yzu = morphisms.ProjectUsingSum(yzuw, yzu)
project_yzuw_to_yuw = morphisms.ProjectUsingSum(yzuw, yuw)
project_yzuw_to_zuw = morphisms.ProjectUsingSum(yzuw, zuw)
project_yzvw_to_yzv = morphisms.ProjectUsingSum(yzvw, yzv)
project_yzvw_to_yvw = morphisms.ProjectUsingSum(yzvw, yvw)
project_yzvw_to_zvw = morphisms.ProjectUsingSum(yzvw, zvw)
project_yuvw_to_yuv = morphisms.ProjectUsingSum(yuvw, yuv)
project_yuvw_to_yuw = morphisms.ProjectUsingSum(yuvw, yuw)
project_yuvw_to_uvw = morphisms.ProjectUsingSum(yuvw, uvw)
project_zuvw_to_zuv = morphisms.ProjectUsingSum(zuvw, zuv)
project_zuvw_to_zuw = morphisms.ProjectUsingSum(zuvw, zuw)
project_zuvw_to_uvw = morphisms.ProjectUsingSum(zuvw, uvw)

# R⁵ -> R⁴ projections
project_xyzuv_to_xyzu = morphisms.ProjectUsingSum(xyzuv, xyzu)
project_xyzuv_to_xyzv = morphisms.ProjectUsingSum(xyzuv, xyzv)
project_xyzuv_to_xyuv = morphisms.ProjectUsingSum(xyzuv, xyuv)
project_xyzuv_to_xzuv = morphisms.ProjectUsingSum(xyzuv, xzuv)
project_xyzuv_to_yzuv = morphisms.ProjectUsingSum(xyzuv, yzuv)
project_xyzuw_to_xyzu = morphisms.ProjectUsingSum(xyzuw, xyzu)
project_xyzuw_to_xyzw = morphisms.ProjectUsingSum(xyzuw, xyzw)
project_xyzuw_to_xyuw = morphisms.ProjectUsingSum(xyzuw, xyuw)
project_xyzuw_to_xzuw = morphisms.ProjectUsingSum(xyzuw, xzuw)
project_xyzuw_to_yzuw = morphisms.ProjectUsingSum(xyzuw, yzuw)
project_xyzvw_to_xyzv = morphisms.ProjectUsingSum(xyzvw, xyzv)
project_xyzvw_to_xyzw = morphisms.ProjectUsingSum(xyzvw, xyzw)
project_xyzvw_to_xyvw = morphisms.ProjectUsingSum(xyzvw, xyvw)
project_xyzvw_to_xzvw = morphisms.ProjectUsingSum(xyzvw, xzvw)
project_xyzvw_to_yzvw = morphisms.ProjectUsingSum(xyzvw, yzvw)
project_xyuvw_to_xyuv = morphisms.ProjectUsingSum(xyuvw, xyuv)
project_xyuvw_to_xyuw = morphisms.ProjectUsingSum(xyuvw, xyuw)
project_xyuvw_to_xyvw = morphisms.ProjectUsingSum(xyuvw, xyvw)
project_xyuvw_to_xuvw = morphisms.ProjectUsingSum(xyuvw, xuvw)
project_xyuvw_to_yuvw = morphisms.ProjectUsingSum(xyuvw, yuvw)
project_xzuvw_to_xzuv = morphisms.ProjectUsingSum(xzuvw, xzuv)
project_xzuvw_to_xzuw = morphisms.ProjectUsingSum(xzuvw, xzuw)
project_xzuvw_to_xzvw = morphisms.ProjectUsingSum(xzuvw, xzvw)
project_xzuvw_to_xuvw = morphisms.ProjectUsingSum(xzuvw, xuvw)
project_xzuvw_to_zuvw = morphisms.ProjectUsingSum(xzuvw, zuvw)
project_yzuvw_to_yzuv = morphisms.ProjectUsingSum(yzuvw, yzuv)
project_yzuvw_to_yzuw = morphisms.ProjectUsingSum(yzuvw, yzuw)
project_yzuvw_to_yzvw = morphisms.ProjectUsingSum(yzuvw, yzvw)
project_yzuvw_to_yuvw = morphisms.ProjectUsingSum(yzuvw, yuvw)
project_yzuvw_to_zuvw = morphisms.ProjectUsingSum(yzuvw, zuvw)

# R⁶ -> R⁵ projections
project_xyzuvw_to_xyzuv = morphisms.ProjectUsingSum(xyzuvw, xyzuv)
project_xyzuvw_to_xyzuw = morphisms.ProjectUsingSum(xyzuvw, xyzuw)
project_xyzuvw_to_xyzvw = morphisms.ProjectUsingSum(xyzuvw, xyzvw)
project_xyzuvw_to_xyuvw = morphisms.ProjectUsingSum(xyzuvw, xyuvw)
project_xyzuvw_to_xzuvw = morphisms.ProjectUsingSum(xyzuvw, xzuvw)
project_xyzuvw_to_yzuvw = morphisms.ProjectUsingSum(xyzuvw, yzuvw)
