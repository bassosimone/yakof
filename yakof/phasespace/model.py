"""
Model building and introspection facilities
===========================================

This module provides a higher-level abstraction for building computational models by:
1. Organizing tensors and enums in dedicated namespaces
2. Managing placeholder naming and enum bindings automatically
3. Providing convenient methods for model construction
4. Supporting type-safe enum conversions between strings and integers

The main class is Model, which acts as a container for:
- A graph namespace containing named tensors
- An enum namespace containing model-specific enumerations
- Methods for binding enums to placeholders
- Utilities for preparing and converting bindings

Example:
    >>> model = Model("service_capacity")
    >>> # Define enum and placeholder
    >>> model.define_placeholder_enum("time", "morning", "evening")
    >>> # Define other placeholders
    >>> model.tensors.capacity = model.placeholder(default_value=50)
    >>> # Define computation
    >>> model.tensors.is_morning = model.tensors.time == model.enums.time.morning
    >>> model.tensors.morning_capacity = tensors.where(
    ...     model.tensors.is_morning,
    ...     model.tensors.capacity,
    ...     model.tensors.capacity * 0.8
    ... )

The module implements three main classes:

1. ModelEnum: Maps string values to integers with type safety
2. GraphNamespace: Contains and manages named tensor nodes
3. Model: Main container class combining enums and tensors

Each class is designed to provide a specific part of the model building
experience while maintaining type safety and clear semantics.

SPDX-License-Identifier: Apache-2.0
"""

from typing import Any, Iterator
import re

from ..backend import graph


# Match Python identifier rules: letter or underscore followed by
# letters, numbers, and underscores.
_enum_name_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


# TODO(bassosimone): consider moving ModelEnum to its own module


class EnumValue(graph.constant):
    """A constant representing an enum value.

    This type exists primarily to make the type system more precise when
    dealing with enum values in the computation graph.
    """


class ModelEnum:
    """Enumeration mapping strings to integers for model placeholders.

    Each enum gets a unique ID (top 16 bits) and can contain up to 2^16
    values (bottom 16 bits). This allows:
    1. Type-safe conversion between strings and integers
    2. Validation that enum values are used with their correct placeholders
    3. Automatic conversion in prepare_bindings()

    The implementation uses bit manipulation to create unique values:
    - enum_id is shifted left by 16 bits to form the high bits
    - value index forms the low 16 bits
    This ensures each enum value across all enums is unique.

    Example:
        >>> time = ModelEnum(0, "morning", "evening")
        >>> time.morning.get_value()  # Returns 0x0000
        >>> time.evening.get_value()  # Returns 0x0001
        >>> weather = ModelEnum(1, "sunny", "rainy")
        >>> weather.sunny.get_value()  # Returns 0x10000
        >>> weather.rainy.get_value()  # Returns 0x10001
    """

    def __init__(self, enum_id: int, *values: str):
        if len(values) >= 1 << 16:
            raise ValueError("an enum cannot contain more than 2^16 values")
        for idx, value in enumerate(values):
            setattr(self, value, EnumValue((enum_id << 16) | idx))

        # Validate all values are valid identifiers
        invalid_values = [
            value for value in values if not _enum_name_pattern.match(value)
        ]
        if invalid_values:
            raise ValueError(
                f"Enum values must be valid Python identifiers. Invalid: {invalid_values}"
                "\nValues should start with a letter or underscore and contain only "
                "letters, numbers, and underscores."
            )

        # Check for duplicates
        if len(set(values)) != len(values):
            seen = set()
            duplicates = [v for v in values if v in seen or seen.add(v)]
            raise ValueError(f"Duplicate enum values not allowed: {duplicates}")

        self.name = ""  # Set by EnumNamespace
        self._values = values

    def __getattr__(self, name: str) -> EnumValue:
        # See the implementation note for Graph.__getattr__
        raise AttributeError(f"No enum value named '{name}' in enum")

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class EnumNamespace:
    """Namespace for enums within a given Model.

    Provides a container for ModelEnum instances with automatic naming.
    When an enum is assigned to this namespace, it automatically gets
    named according to its attribute name.

    Example:
        >>> model = Model()
        >>> model.enums.time = model.enum("morning", "evening")
        >>> model.enums.time.name  # Returns "time"
    """

    def __init__(self):
        self._enums: list[ModelEnum] = []

    def __setattr__(self, name: str, value: ModelEnum):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        value.name = name  # Give the enum a proper name
        self._enums.append(value)
        self.__dict__[name] = value

    def __getattr__(self, name: str) -> ModelEnum:
        # See the implementation note for Graph.__getattr__
        raise AttributeError(f"No enum value named '{name}' in enum")


# TODO(bassosimone): consider using backend.oriented.Field to implement
# part of the GraphNamespace functionality


class GraphNamespace:
    """Namespace containing named tensor nodes.

    Design notes:
    1. Names are automatically assigned when tensors are added
    2. Names must be unique within the namespace
    3. Implements TensorGraph protocol for evaluation
    4. Maintains definition order for deterministic evaluation
    5. Provides type hints through __getattr__

    Example:
        >>> model = Model()
        >>> model.graph.x = model.placeholder()  # x.name becomes "x"
        >>> model.graph.y = model.placeholder()  # y.name becomes "y"
        >>> model.graph.sum = model.graph.x + model.graph.y  # sum.name becomes "sum"
    """

    def __init__(self):
        self._tensors: list[graph.Tensor] = []

    def __setattr__(self, name: str, value: graph.Tensor):
        """Assigns a name to a tensor and ensures naming consistency."""
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        if name in self.__dict__:
            raise ValueError(f"field already exists: {name}")

        if isinstance(value, graph.placeholder) and value.name and value.name != name:
            raise ValueError(
                f"Placeholder name mismatch: {value.name} != {name}. "
                f"Use model.placeholder() for automatic naming."
            )

        value.name = name
        self.__dict__[name] = value
        self._tensors.append(value)

    def __getattr__(self, name: str) -> graph.Tensor:
        # Implementation note: this method seems unnecessary but actually it's
        # paramount to keep it, since it tells the type system that in this class
        # we have arbitrary attributes and that when there's an unknown attrib
        # its type is actually `graph.Tensor`. This kind of oddities is probably
        # the main reason why I continue programming and have not gave in to
        # my underlying desire to work as a clerk for a gas station pump. ("Tra
        # un anno a Coachella, e tra due anni a fare il benzinaio!".)
        raise AttributeError(f"No tensor named '{name}' in graph")

    # There are two ways to iterate: through method implementing the
    # numpy_backend.TensorGraph protocol and via __iter__

    def iterable_graph(self) -> Iterator[graph.Tensor]:
        """Returns an iterator over the tensors in the order in which they
        were defined, thus allowing easy and direct evaluation.

        Implements, e.g., the numpy_backend.TensorGraph protocol."""
        return iter(self._tensors)

    def __iter__(self) -> Iterator[graph.Tensor]:
        return iter(self._tensors)


class Model:
    """A model containing a graph of tensors and extra information
    required to fully instantiate and use the model.

    The Model class provides:
    1. Namespaces for organizing tensors and enums
    2. Methods for creating and binding enums to placeholders
    3. Utilities for preparing bindings with automatic conversions
    4. Helper methods for common model building patterns

    Example:
        >>> m = model.Model("service model")
        >>> m.enums.time = m.enum("morning", "evening")  # Define enum
        >>> m.graph.time = m.placeholder()  # Create placeholder
        >>> m.bind_enum(m.graph.time, m.enums.time)  # Bind enum to placeholder
        >>> m.graph.customers = m.placeholder()
        >>> m.graph.service_time = m.placeholder(default_value=15.0)
        >>> m.graph.service_rate = m.graph.customers / m.graph.service_time
        >>> m.graph.capacity = m.placeholder(default_value=50)
        >>> m.graph.sustainable = m.graph.service_rate <= m.graph.capacity
        >>> for tensor in m.graph:
        >>>     print(tensor)
    """

    def __init__(self, name: str = ""):
        self.enums = EnumNamespace()
        self.tensors = GraphNamespace()
        self.name = name
        self._enum_id = 0
        self._enum_bindings: dict[str, ModelEnum] = {}

    def __repr__(self):
        return f"{self.name}"

    def get_tensor_by_name(self, name: str) -> graph.Tensor:
        """Get a tensor by name."""
        return getattr(self.tensors, name)

    def enum(self, *values: str) -> ModelEnum:
        """Create a new enumeration for this model.

        Args:
            *values: The string values for this enumeration

        Returns:
            A ModelEnum that can be assigned to the model's enum namespace

        Example:
            >>> m = Model()
            >>> m.enums.time = m.enum("morning", "evening")
            >>> isinstance(m.enums.time.morning, graph.Tensor)  # True
        """
        enum_id = self._enum_id
        self._enum_id += 1
        return ModelEnum(enum_id, *values)

    def bind_enum(self, placeholder: graph.placeholder, enum: ModelEnum) -> None:
        """Bind an enum to a placeholder.

        After binding, the placeholder will accept string values that will be
        automatically converted to the corresponding enum integer values.

        Args:
            placeholder: The placeholder to bind the enum to
            enum: The enum to bind

        Raises:
            ValueError: If the placeholder isn't named yet (not added to graph)
                      or if it's already bound to a different enum

        Example:
            >>> m = Model()
            >>> m.enums.time = m.enum("morning", "evening")
            >>> m.graph.time = m.placeholder()
            >>> m.bind_enum(m.graph.time, m.enums.time)
            >>> # Now bindings can use strings:
            >>> bindings = {"time": "morning"}  # Automatically converted
        """
        if not placeholder.name:
            raise ValueError("Cannot bind enum to unnamed placeholder")
        if placeholder.name in self._enum_bindings:
            raise ValueError(f"Placeholder {placeholder.name} already bound to an enum")
        if placeholder.default_value is not None:
            raise ValueError("Cannot bind enum to placeholder with default_value")
        if placeholder.dtype is not None and placeholder.dtype not in (
            graph.DType.INT32,
            graph.DType.INT64,
        ):
            raise ValueError(
                "Cannot bind enum to placeholder with explicit dtype that is neither INT32 nor INT64"
            )
        self._enum_bindings[placeholder.name] = enum

    def prepare_bindings(
        self, bindings: dict[str, Any]
    ) -> dict[str, graph.ScalarValue]:
        """Convert external bindings to internal representation.

        Handles string values for enum-bound placeholders by converting them
        to their corresponding integer values.

        Args:
            bindings: External bindings with possibly string enum values

        Returns:
            Bindings with all values converted to their internal representation

        Example:
            >>> m = Model()
            >>> # ... setup model with enum ...
            >>> bindings = m.prepare_bindings({
            ...     "time": "morning",  # String gets converted to int
            ...     "customers": 42     # Non-enum values pass through
            ... })
        """
        result = {}
        for name, value in bindings.items():
            if name in self._enum_bindings and isinstance(value, str):
                enum = self._enum_bindings[name]
                result[name] = getattr(enum, value).get_value()
            else:
                result[name] = value
        return result

    def placeholder(
        self,
        default_value: graph.ScalarValue | None = None,
        dtype: graph.DType | None = None,
        name: str = "",
        description: str = "",
        unit: str = "",
    ) -> graph.placeholder:
        """Create a placeholder with optional default value and dtype.

        The placeholder's name will be automatically set when it's assigned
        to a model attribute.

        Args:
            default_value: Optional default value for the placeholder
            dtype: Optional specific dtype (inferred if not provided)
            name: Optional name for documentation
            description: Optional description for documentation
            unit: Optional unit for documentation

        Returns:
            A placeholder tensor that can be assigned to the model's graph

        Example:
            >>> m = model.Model("my model")
            >>> m.graph.rate = m.placeholder(
            ...     default_value=1.0,
            ...     description="Service rate",
            ...     unit="customers/hour"
            ... )
            >>> print(m.graph.rate.name)  # prints: 'rate'

        Note: if you want to use the placeholder as an enum, you should
        ensure its type is set to graph.DType.INT32 or .INT64. For example:
            >>> m = model.Model()
            >>> m.graph.time = m.placeholder(dtype=graph.DType.INT64)
            >>> # Now it can be bound to an enum
        """
        # TODO(bassosimone): forward name, description and unit to the
        # tensor itself. Also, consider whether there are other factories
        # that could benfit from this kind of info forwarding.
        return graph.placeholder(name="", dtype=dtype, default_value=default_value)

    def placeholders(self) -> list[graph.Tensor]:
        """Returns the list of all the model placeholders.

        This is useful for:
        1. Inspecting what inputs a model needs
        2. Validating bindings cover all required inputs
        3. Generating documentation or UI elements
        """
        return [
            tensor for tensor in self.tensors if isinstance(tensor, graph.placeholder)
        ]

    def define_placeholder(
        self,
        name: str,
        default_value: graph.ScalarValue | None = None,
        dtype: graph.DType | None = None,
    ) -> graph.Tensor:
        """Define a placeholder in the model's graph.

        Equivalent to:
            >>> m.graph.name = m.placeholder()

        This is a convenience method that combines creation and assignment
        in one step.

        Example:
            >>> m.define_placeholder("foobar")
            >>> m.graph.foobar  # Access through graph namespace
        """
        ph = self.placeholder(default_value, dtype)
        setattr(self.tensors, name, ph)
        return ph

    def define_placeholder_enum(self, name: str, *values: str) -> graph.Tensor:
        """Define a placeholder and its corresponding enum.

        Creates and binds both the placeholder and enum in one operation.
        This is a convenience method that combines several operations:
        1. Creates an enum with the given values
        2. Creates a placeholder with appropriate type
        3. Adds both to their respective namespaces
        4. Binds them together

        Equivalent to:
            >>> m.graph.name = m.placeholder(dtype=graph.DType.INT64)
            >>> m.enums.name = m.enum("foo", "bar", "baz")
            >>> m.bind_enum(m.graph.name, m.enums.name)

        Example:
            >>> m.define_placeholder_enum("time", "morning", "evening")
            >>> m.graph.time == m.enums.time.morning  # Use through namespaces
            >>> # Now bindings can use strings:
            >>> bindings = {"time": "morning"}  # Will be converted automatically
        """
        ph = self.placeholder(dtype=graph.DType.INT64)
        setattr(self.tensors, name, ph)
        enum = self.enum(*values)
        setattr(self.enums, name, enum)
        self.bind_enum(ph, enum)
        return ph
