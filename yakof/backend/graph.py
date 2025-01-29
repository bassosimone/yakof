"""
Computation graph building
==========================

This module allows to build an abstract computation graph using TensorFlow-like
computation primitives and concepts. These primitives and concepts are similar to
NumPy primitives, but we picked up TensorFlow ones when they disagree, since it
seems that TensorFlow uses better naming, probably because it is a more recent
library. Our general goal, in any case is that of emulating TensorFlow as much as
possible and there are some deprecated names that diverge from that standard.

Here's an example of what you can do with this module:

    >>> x = graph.constant(1.0)
    >>> y = graph.add(x, 2.0)
    >>> z = graph.multiply(y, 3.0)
    >>> k = x + y * z

Note how it is possible to express operations both using a functional style, i.e.,
by calling `.add`, and by using infix operators, i.e., by using `+`. We did this to
facilitate writing equations describing sustainability models.

However, because our goal is to *capture* the arguments provided to such functions
for deferred evaluation, we are using classes instead of functions. To keep the
usage as similar as possible to TensorFlow, the classes are named using snake_case
rather than CamelCase. This is a pragmatic and conscious choice, which implies
we are breaking PEP8, that results in easier to read and reason about code.

The main type in this module is the `Tensor` class, which represents a node in the
computation graph. This class supports arithmetic, comparison, and logical operators
by overriding the infix operators. This allows for a more natural and readable
representation of the computation graph. Also, we subclass this class as needed
to represent all the possible operations that we support.

The general goal of a `Tensor` subclass, in fact, is to freeze the provided
arguments and to provide a `dtype` property that returns the data type of the
result of the operation. Because the shape depends on information provided
at evaluation time, this implementation does not track shapes, except in cases
where the shape is known at construction time.

Overall, this implementation is probably conceptually very similar to the
TensorFlow 1.x lazy API for constructing computation graphs.

As a result, we also use the `placeholder` concept for representing inputs for
the computation graph deferred evaluation, just like in TensorFlow.

Additionally, this module contains helper functions to introduce the evaluation of
specific probability distributions as nodes of the graph. These functions are not
part of TensorFlow or NumPy API exactly as they are used here. However, we pragmatically
decided that supporting them as functions was the most convenient choice for us. For
this functionality we loosely model our API on `scipy.stats`.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence, Union, TypeVar, Any
from typing_extensions import deprecated

import warnings

# Basic types
Shape = tuple[int, ...]
Axis = Union[int, Sequence[int]]
ScalarValue = Union[float, int, bool]

# Type variable for scalar value extraction
T = TypeVar("T", float, int, bool)


class DType(Enum):
    """Supported data types, ordered by type promotion precedence."""

    BOOL = "bool"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    def __lt__(self, other: DType) -> bool:
        """Enable comparison for type promotion."""
        dtype_order = {
            DType.BOOL: 0,
            DType.INT32: 1,
            DType.INT64: 2,
            DType.FLOAT32: 3,
            DType.FLOAT64: 4,
        }
        return dtype_order[self] < dtype_order[other]


def promote_dtypes(dt1: DType, dt2: DType) -> DType:
    """Determine result type of operation between two dtypes."""
    return max(dt1, dt2)


class Tensor:
    """
    Base class for all tensor nodes in the computation graph.

    Implementation note: we override __eq__ with incompatible types to support
    lazy tensor equality construction. As a result, we're forced to implement
    a __hash__ method using the object's id to support adding to sets and dicts.

    The tensor name is used for:

    1. Debugging and visualization purposes.

    2. To assign a name to the tensor in the computation graph model,
    which is implemented in the model.py module.
    """

    def __init__(self, name: str = "", description: str = ""):
        self._id = id(self)
        self.name = name
        self.description = description

    def __hash__(self) -> int:
        return self._id

    def get_description(self) -> str:
        """Return a description of the tensor."""
        return self.description

    @property
    def dtype(self) -> DType:
        raise NotImplementedError()  # implemented by subclasses

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}')"

    # Arithmetic operators
    def __add__(self, other: Tensor | ScalarValue) -> Tensor:
        return add(self, other)

    def __radd__(self, other: Tensor | ScalarValue) -> Tensor:
        return add(other, self)

    def __sub__(self, other: Tensor | ScalarValue) -> Tensor:
        return subtract(self, other)

    def __rsub__(self, other: Tensor | ScalarValue) -> Tensor:
        return subtract(other, self)

    def __mul__(self, other: Tensor | ScalarValue) -> Tensor:
        return multiply(self, other)

    def __rmul__(self, other: Tensor | ScalarValue) -> Tensor:
        return multiply(other, self)

    def __truediv__(self, other: Tensor | ScalarValue) -> Tensor:
        return divide(self, other)

    def __rtruediv__(self, other: Tensor | ScalarValue) -> Tensor:
        return divide(other, self)

    # Comparison operators
    def __eq__(self, other: Tensor | ScalarValue) -> Tensor:  # type: ignore
        return equal(self, other)

    def __ne__(self, other: Tensor | ScalarValue) -> Tensor:  # type: ignore
        return not_equal(self, other)

    def __lt__(self, other: Tensor | ScalarValue) -> Tensor:
        return less(self, other)

    def __le__(self, other: Tensor | ScalarValue) -> Tensor:
        return less_equal(self, other)

    def __gt__(self, other: Tensor | ScalarValue) -> Tensor:
        return greater(self, other)

    def __ge__(self, other: Tensor | ScalarValue) -> Tensor:
        return greater_equal(self, other)

    # Logical operators
    def __and__(self, other: Tensor | ScalarValue) -> Tensor:
        return logical_and(self, other)

    def __rand__(self, other: Tensor | ScalarValue) -> Tensor:
        return logical_and(other, self)

    def __or__(self, other: Tensor | ScalarValue) -> Tensor:
        return logical_or(self, other)

    def __ror__(self, other: Tensor | ScalarValue) -> Tensor:
        return logical_or(other, self)

    def __xor__(self, other: Tensor | ScalarValue) -> Tensor:
        return logical_xor(self, other)

    def __rxor__(self, other: Tensor | ScalarValue) -> Tensor:
        return logical_xor(other, self)

    def __invert__(self) -> Tensor:
        return logical_not(self)


class constant(Tensor):
    """
    A constant value in the computation graph.

    BUG: this implementation only allows adding scalar constant. To provide
    specific arrays (e.g., an `np.ndarray`) as constant values, one should
    use a `placeholder` and provide the array at evaluation time.
    """

    def __init__(self, value: ScalarValue, shape: Shape | None = None):
        super().__init__()
        self.value = value
        self.shape = shape
        self._dtype = self._infer_dtype(value)

    def _infer_dtype(self, value: ScalarValue) -> DType:
        # Implementation note: we assume we're on 64 bit systems
        if isinstance(value, bool):
            return DType.BOOL
        if isinstance(value, int):
            return DType.INT64
        return DType.FLOAT64

    @property
    def dtype(self) -> DType:
        return self._dtype

    def get_value(self) -> ScalarValue:
        """Get the constant's value."""
        return self.value

    def __repr__(self) -> str:
        return f"constant({self.value})"


def ensure_tensor(value: Tensor | ScalarValue) -> Tensor:
    """Convert a scalar value to a tensor if it isn't one already."""
    if isinstance(value, ScalarValue):
        value = constant(value)
    return value


class placeholder(Tensor):
    """A placeholder for input values with optional scalar default value."""

    def __init__(
        self,
        name: str = "",
        dtype: DType | None = None,
        default_value: Tensor | ScalarValue | None = None,
    ):
        super().__init__(name)
        self._dtype = dtype or DType.FLOAT64
        self.default_value = (
            ensure_tensor(default_value) if default_value is not None else None
        )

    @property
    def dtype(self) -> DType:
        return self._dtype


class binary_op(Tensor):
    """Base class for binary operations with broadcasting."""

    def __init__(self, left: Tensor, right: Tensor):
        super().__init__()
        self.left = left
        self.right = right


class add(binary_op):
    """Element-wise addition of two tensors."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        super().__init__(ensure_tensor(x), ensure_tensor(y))

    @property
    def dtype(self) -> DType:
        return promote_dtypes(self.left.dtype, self.right.dtype)


class subtract(binary_op):
    """Element-wise subtraction of two tensors."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        super().__init__(ensure_tensor(x), ensure_tensor(y))

    @property
    def dtype(self) -> DType:
        return promote_dtypes(self.left.dtype, self.right.dtype)


class multiply(binary_op):
    """Element-wise multiplication of two tensors."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        super().__init__(ensure_tensor(x), ensure_tensor(y))

    @property
    def dtype(self) -> DType:
        return promote_dtypes(self.left.dtype, self.right.dtype)


class divide(binary_op):
    """Element-wise division of two tensors."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        super().__init__(ensure_tensor(x), ensure_tensor(y))

    @property
    def dtype(self) -> DType:
        return DType.FLOAT64  # Division always produces float


class comparison_op(binary_op):
    """Base class for comparison operations."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        super().__init__(ensure_tensor(x), ensure_tensor(y))

    @property
    def dtype(self) -> DType:
        return DType.BOOL


class equal(comparison_op):
    """Element-wise equality comparison."""


class not_equal(comparison_op):
    """Element-wise inequality comparison."""


class less(comparison_op):
    """Element-wise less than comparison."""


class less_equal(comparison_op):
    """Element-wise less than or equal comparison."""


class greater(comparison_op):
    """Element-wise greater than comparison."""


class greater_equal(comparison_op):
    """Element-wise greater than or equal comparison."""


class logical_op(binary_op):
    """Base class for logical operations on boolean tensors."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        x, y = ensure_tensor(x), ensure_tensor(y)
        if x.dtype != DType.BOOL or y.dtype != DType.BOOL:
            raise ValueError(
                f"Logical operation requires bool operands, got {x.dtype} and {y.dtype}"
            )
        super().__init__(x, y)

    @property
    def dtype(self) -> DType:
        return DType.BOOL


class logical_and(logical_op):
    """Element-wise logical AND."""


class logical_or(logical_op):
    """Element-wise logical OR."""


class logical_xor(logical_op):
    """Element-wise logical XOR."""


class logical_not(Tensor):
    """Element-wise logical NOT."""

    def __init__(self, x: Tensor | ScalarValue):
        super().__init__()
        self.x = ensure_tensor(x)
        if self.x.dtype != DType.BOOL:
            raise ValueError(f"Logical not requires bool operand, got {self.x.dtype}")

    @property
    def dtype(self) -> DType:
        return DType.BOOL


class reshape(Tensor):
    """Reshape tensor to new dimensions."""

    def __init__(self, x: Tensor | ScalarValue, shape: Shape):
        super().__init__()
        self.x = ensure_tensor(x)
        self.new_shape = shape

    @property
    def dtype(self) -> DType:
        return self.x.dtype


class expand_dims(Tensor):
    """Insert new axis of size 1."""

    def __init__(self, x: Tensor | ScalarValue, axis: Axis):
        super().__init__()
        self.x = ensure_tensor(x)
        self.axis = axis

    @property
    def dtype(self) -> DType:
        return self.x.dtype


class squeeze(Tensor):
    """Remove axes of size 1."""

    def __init__(self, x: Tensor | ScalarValue, axis: Axis | None = None):
        super().__init__()
        self.x = ensure_tensor(x)
        self.axes = axis

    @property
    def dtype(self) -> DType:
        return self.x.dtype


class reduce_sum(Tensor):
    """Sum tensor elements along specified axes."""

    def __init__(self, x: Tensor | ScalarValue, axis: Axis | None = None):
        super().__init__()
        self.x = ensure_tensor(x)
        self.axes = axis

    @property
    def dtype(self) -> DType:
        return self.x.dtype


class where(Tensor):
    """Select elements from x or y based on condition."""

    def __init__(
        self,
        condition: Tensor | ScalarValue,
        x: Tensor | ScalarValue,
        y: Tensor | ScalarValue,
    ):
        super().__init__()
        self.condition = ensure_tensor(condition)
        if self.condition.dtype != DType.BOOL:
            warnings.warn(f"Where condition must be bool, got {self.condition.dtype}")
        self.x = ensure_tensor(x)
        self.y = ensure_tensor(y)
        self._dtype = promote_dtypes(self.x.dtype, self.y.dtype)

    @property
    def dtype(self) -> DType:
        return self._dtype


class multi_clause_where(Tensor):
    """Conditional expression, similar to Scheme's cond. Allows one to write
    complex conditional expressions more naturally than where.

    Args:
        *cases: List of (condition, value) pairs, where the last condition
              should be True (explicit else clause).

    Example:
        >>> service_capacity = graph.multi_clause_where(
        ...     (time == MORNING & weather == SUNNY, 6),
        ...     (time == MORNING & weather == RAINY, 5),
        ...     (time == EVENING & weather == SUNNY, 4),
        ...     (True, 3)  # else clause
        ... )

    Raises:
        ValueError: If no True condition is provided as last case
                   (would make the function partial)
    """

    def __init__(
        self,
        *cases: tuple[Tensor | ScalarValue, Tensor | ScalarValue],
    ):
        super().__init__()
        if not cases:
            raise ValueError("multi_clause_where requires at least one case")

        # Build the list of tensored cases
        self.cases = [
            (ensure_tensor(cond), ensure_tensor(value)) for cond, value in cases
        ]

        # Ensure last condition is True
        last_cond, _ = self.cases[-1]
        if not (
            isinstance(last_cond, constant)
            and isinstance(last_cond.get_value(), bool)
            and last_cond.get_value() is True
        ):
            raise ValueError("Last condition must be True")

        # Validate all conditions are boolean
        for cond, _ in self.cases[:-1]:  # Skip last (True) condition
            if cond.dtype != DType.BOOL:
                raise ValueError(f"Condition must be boolean, got {cond.dtype}")

    @property
    def dtype(self) -> DType:
        return promote_dtypes(*(value.dtype for _, value in self.cases))


@deprecated(
    "Not consistent with TensorFlow API; please, use multi_clause_where instead"
)
class cond(multi_clause_where):
    pass


class uniform_rvs(Tensor):
    """Generate uniform random values."""

    def __init__(
        self,
        shape: Shape,
        loc: Tensor | ScalarValue = 0.0,
        scale: Tensor | ScalarValue = 1.0,
    ):
        super().__init__()
        self.shape = shape
        self.loc = ensure_tensor(loc)
        self.scale = ensure_tensor(scale)

    @property
    def dtype(self) -> DType:
        return DType.FLOAT64


class uniform_cdf(Tensor):
    """Compute CDF of uniform distribution."""

    def __init__(
        self,
        x: Tensor | ScalarValue,
        loc: Tensor | ScalarValue = 0.0,
        scale: Tensor | ScalarValue = 1.0,
    ):
        super().__init__()
        self.x = ensure_tensor(x)
        self.loc = ensure_tensor(loc)
        self.scale = ensure_tensor(scale)

    @property
    def dtype(self) -> DType:
        return DType.FLOAT64


class normal_rvs(Tensor):
    """Generate normal (Gaussian) random values."""

    def __init__(
        self,
        shape: Shape,
        loc: Tensor | ScalarValue = 0.0,
        scale: Tensor | ScalarValue = 1.0,
    ):
        super().__init__()
        self.shape = shape
        self.loc = ensure_tensor(loc)
        self.scale = ensure_tensor(scale)

    @property
    def dtype(self) -> DType:
        return DType.FLOAT64


class normal_cdf(Tensor):
    """Compute CDF of normal distribution."""

    def __init__(
        self,
        x: Tensor | ScalarValue,
        loc: Tensor | ScalarValue = 0.0,
        scale: Tensor | ScalarValue = 1.0,
    ):
        super().__init__()
        self.x = ensure_tensor(x)
        self.loc = ensure_tensor(loc)
        self.scale = ensure_tensor(scale)

    @property
    def dtype(self) -> DType:
        return DType.FLOAT64


class maximum(Tensor):
    """Element-wise maximum of two tensors."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        super().__init__()
        self.x = ensure_tensor(x)
        self.y = ensure_tensor(y)

    @property
    def dtype(self) -> DType:
        return promote_dtypes(self.x.dtype, self.y.dtype)


class reduce_mean(Tensor):
    """Mean reduction along specified axis.

    This operation computes the mean of elements across dimensions given by axis.
    For example, if x has shape (A, B, C) and axis=1, the output will have
    shape (A, C), containing means across dimension B.

    Args:
        x: The tensor to reduce
        axis: The axis or axes along which to compute mean.
            If None, reduces over all dimensions.
    """

    def __init__(self, x: Tensor | ScalarValue, axis: Axis | None = None):
        super().__init__()
        self.x = ensure_tensor(x)
        self.axis = axis

    @property
    def dtype(self) -> DType:
        # Mean of integers should give float
        if self.x.dtype in (DType.INT32, DType.INT64):
            return DType.FLOAT64
        return self.x.dtype


class exp(Tensor):
    """Element-wise exponential."""

    def __init__(self, x: Tensor | ScalarValue):
        super().__init__()
        self.x = ensure_tensor(x)

    @property
    def dtype(self) -> DType:
        return self.x.dtype


class power(Tensor):
    """Element-wise power function."""

    def __init__(self, x: Tensor | ScalarValue, y: Tensor | ScalarValue):
        super().__init__()
        self.x = ensure_tensor(x)
        self.y = ensure_tensor(y)

    @property
    def dtype(self) -> DType:
        return promote_dtypes(self.x.dtype, self.y.dtype)


class pow(power):
    """Alias for power for compatibility with NumPy."""


class log(Tensor):
    """Element-wise natural logarithm."""

    def __init__(self, x: Tensor | ScalarValue):
        super().__init__()
        self.x = ensure_tensor(x)

    @property
    def dtype(self) -> DType:
        return self.x.dtype
